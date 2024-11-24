// Import dependencies
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest';
import { Chart } from 'https://cdn.jsdelivr.net/npm/chart.js';

// Global state management
const state = {
    currentMode: 'iid',
    isLoading: false,
    clients: Array(5).fill().map((_, i) => ({
        id: i + 1,
        data: null,
        model: null,
        trainHistory: [],
        currentAccuracy: 0,
        isTraining: false,
        hyperparameters: {
            learningRate: 0.01,
            epochs: 5,
            batchSize: 32
        }
    })),
    globalModel: null,
    globalTestSet: {
        x: null,
        y: null
    },
    dropoutProbability: 1.0,
    charts: {
        clientCharts: [],
        globalChart: null
    }
};

// UI Management
const UI = {
    setLoading: (isLoading) => {
        state.isLoading = isLoading;
        document.body.classList.toggle('loading', isLoading);
        document.querySelectorAll('button').forEach(btn => {
            btn.disabled = isLoading;
        });
    },

    showError: (message) => {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 3000);
    },

    updateSliderValue: (slider, value) => {
        const valueDisplay = slider.nextElementSibling;
        const decimals = slider.step.includes('.') ? 
            slider.step.split('.')[1].length : 0;
        valueDisplay.textContent = Number(value).toFixed(decimals);
    },

    updateClientStatus: (clientId, status) => {
        const clientCard = document.querySelector(`#client${clientId}`);
        ['loading', 'error', 'training'].forEach(s => {
            clientCard.classList.toggle(s, s === status);
        });
    },

    updateAccuracy: (clientId, accuracy) => {
        const accuracyElement = document.querySelector(`#client${clientId} .accuracy`);
        if (accuracyElement) {
            accuracyElement.textContent = `Acc: ${(accuracy * 100).toFixed(1)}%`;
            accuracyElement.classList.add('updated');
            setTimeout(() => accuracyElement.classList.remove('updated'), 300);
        }
    }
};

// Data Management
class DataManager {
    static async loadClientData(clientId, mode) {
        try {
            UI.updateClientStatus(clientId, 'loading');
            const response = await fetch(`data/${mode}/client${clientId}.json`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            
            // Validate data structure
            if (!this.validateDataStructure(data)) {
                throw new Error('Invalid data structure');
            }
            
            // Convert to tensors and normalize
            const processedData = {
                train: {
                    x: tf.tensor2d(data.x_train).div(255.0),
                    y: tf.tensor1d(data.y_train).oneHot(10)
                },
                test: {
                    x: tf.tensor2d(data.x_test).div(255.0),
                    y: tf.tensor1d(data.y_test).oneHot(10)
                }
            };
            
            UI.updateClientStatus(clientId, null);
            return processedData;
        } catch (error) {
            UI.updateClientStatus(clientId, 'error');
            UI.showError(`Failed to load data for client ${clientId}: ${error.message}`);
            throw error;
        }
    }

    static validateDataStructure(data) {
        return data.x_train && data.y_train && data.x_test && data.y_test &&
               data.x_train.length === 1000 && data.x_train[0].length === 784 &&
               data.y_train.length === 1000 &&
               data.x_test.length === 100 && data.x_test[0].length === 784 &&
               data.y_test.length === 100;
    }

    static async initializeGlobalTestSet(mode) {
        const allTestX = [];
        const allTestY = [];
        
        try {
            for (let i = 1; i <= 5; i++) {
                const response = await fetch(`data/${mode}/client${i}.json`);
                const data = await response.json();
                allTestX.push(...data.x_test);
                allTestY.push(...data.y_test);
            }
            
            state.globalTestSet.x = tf.tensor2d(allTestX).div(255.0);
            state.globalTestSet.y = tf.tensor1d(allTestY).oneHot(10);
        } catch (error) {
            UI.showError('Failed to initialize global test set');
            throw error;
        }
    }
}

// Model Architecture
class SimpleMLP {
    static create() {
        const model = tf.sequential();
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            inputShape: [784]
        }));
        model.add(tf.layers.dense({
            units: 10,
            activation: 'softmax'
        }));
        
        return model;
    }
}

// Training Manager
class TrainingManager {
    static async trainLocalModel(clientId) {
        const client = state.clients[clientId - 1];
        if (!client.data) {
            UI.showError(`No data available for client ${clientId}`);
            return;
        }

        try {
            UI.updateClientStatus(clientId, 'training');
            client.isTraining = true;

            const model = client.model || SimpleMLP.create();
            
            model.compile({
                optimizer: tf.train.adam(client.hyperparameters.learningRate),
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });

            const history = await model.fit(
                client.data.train.x,
                client.data.train.y,
                {
                    epochs: client.hyperparameters.epochs,
                    batchSize: client.hyperparameters.batchSize,
                    validationData: [client.data.test.x, client.data.test.y],
                    callbacks: {
                        onEpochEnd: (epoch, logs) => {
                            ChartManager.updateClientChart(clientId, logs);
                            client.currentAccuracy = logs.val_accuracy;
                            UI.updateAccuracy(clientId, logs.val_accuracy);
                        }
                    }
                }
            );

            client.model = model;
            client.trainHistory.push(history);
            
        } catch (error) {
            UI.showError(`Training failed for client ${clientId}: ${error.message}`);
        } finally {
            client.isTraining = false;
            UI.updateClientStatus(clientId, null);
        }
    }

    static async aggregateModels() {
        const participatingClients = state.clients.filter(
            () => Math.random() < state.dropoutProbability
        );

        if (participatingClients.length === 0) {
            UI.showError('No clients participating due to dropout!');
            return;
        }

        try {
            UI.setLoading(true);

            // Get weights from all participating clients
            const weights = participatingClients.map(client => 
                client.model.getWeights().map(w => w.clone())
            );

            // Average weights
            const averagedWeights = weights[0].map((_, layerIdx) => {
                const layerWeights = weights.map(w => w[layerIdx]);
                return tf.tidy(() => {
                    const sum = tf.addN(layerWeights);
                    return sum.div(weights.length);
                });
            });

            // Update global model
            state.globalModel = SimpleMLP.create();
            state.globalModel.setWeights(averagedWeights);

            // Evaluate global model
            const accuracy = await this.evaluateGlobalModel();
            ChartManager.updateGlobalChart(accuracy);

        } catch (error) {
            UI.showError('Model aggregation failed: ' + error.message);
        } finally {
            UI.setLoading(false);
        }
    }

    static async evaluateGlobalModel() {
        const result = await state.globalModel.evaluate(
            state.globalTestSet.x,
            state.globalTestSet.y
        );
        return result[1]; // accuracy
    }
}

// Chart Management
class ChartManager {
    static initializeCharts() {
        // Initialize client charts
        state.charts.clientCharts = Array(5).fill().map((_, i) => 
            new Chart(
                document.querySelector(`#client${i+1} .plot-container`).getContext('2d'),
                this.createChartConfig(`Client ${i+1} Training Progress`)
            )
        );
        
        // Initialize global chart
        state.charts.globalChart = new Chart(
            document.querySelector('.global-plot').getContext('2d'),
            this.createChartConfig('Global Model Accuracy')
        );
    }

    static createChartConfig(label) {
        return {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: label,
                    data: [],
                    borderColor: '#9B4DCA',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        };
    }

    static updateClientChart(clientId, logs) {
        const chart = state.charts.clientCharts[clientId - 1];
        chart.data.labels.push(logs.epoch + 1);
        chart.data.datasets[0].data.push(logs.val_accuracy * 100);
        chart.update();
    }

    static updateGlobalChart(accuracy) {
        const chart = state.charts.globalChart;
        chart.data.labels.push(chart.data.labels.length + 1);
        chart.data.datasets[0].data.push(accuracy * 100);
        chart.update();
    }

    static resetCharts() {
        state.charts.clientCharts.forEach(chart => {
            chart.data.labels = [];
            chart.data.datasets[0].data = [];
            chart.update();
        });
        
        state.charts.globalChart.data.labels = [];
        state.charts.globalChart.data.datasets[0].data = [];
        state.charts.globalChart.update();
    }
}

// Event Handlers
async function initializeApp() {
    try {
        UI.setLoading(true);
        ChartManager.initializeCharts();
        await DataManager.initializeGlobalTestSet('iid');
        for (let i = 1; i <= 5; i++) {
            state.clients[i-1].data = await DataManager.loadClientData(i, 'iid');
        }
    } catch (error) {
        UI.showError('Failed to initialize application: ' + error.message);
    } finally {
        UI.setLoading(false);
    }
}

function bindEventListeners() {
    // Mode selection
    document.querySelectorAll('.mode-button').forEach(button => {
        button.addEventListener('click', async (e) => {
            const newMode = e.target.textContent.toLowerCase();
            if (newMode !== state.currentMode && !state.isLoading) {
                try {
                    UI.setLoading(true);
                    state.currentMode = newMode;
                    ChartManager.resetCharts();
                    await DataManager.initializeGlobalTestSet(newMode);
                    for (let i = 1; i <= 5; i++) {
                        state.clients[i-1].data = await DataManager.loadClientData(i, newMode);
                    }
                } catch (error) {
                    UI.showError('Failed to switch modes: ' + error.message);
                } finally {
                    UI.setLoading(false);
                }
            }
        });
    });

    // Client training buttons
    document.querySelectorAll('.client-train-button').forEach(button => {
        button.addEventListener('click', async (e) => {
            const clientId = parseInt(e.target.closest('.client-card').id.replace('client', ''));
            if (!state.clients[clientId-1].isTraining) {
                await TrainingManager.trainLocalModel(clientId);
            }
        });
    });

    // Global training button
    document.querySelector('.train-all-button').addEventListener('click', async () => {
        for (let i = 1; i <= 5; i++) {
            if (!state.clients[i-1].isTraining) {
                await TrainingManager.trainLocalModel(i);
            }
        }
    });

    // Aggregation button
    document.querySelector('.aggregate-button').addEventListener('click', 
        async () => await TrainingManager.aggregateModels()
    );

    // Hyperparameter sliders
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.addEventListener('input', (e) => {
            const clientCard = e.target.closest('.client-card');
            const value = parseFloat(e.target.value);
            
            UI.updateSliderValue(e.target, value);
            
            if (clientCard) {
                const clientId = parseInt(clientCard.id.replace('client', ''));
                const paramName = e.target.getAttribute('data-param');
                state.clients[clientId-1].hyperparameters[paramName] = value;
            } else {
                state.dropoutProbability = value;
            }
        });
    });
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    bindEventListeners();
});