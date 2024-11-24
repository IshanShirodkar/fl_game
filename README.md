# 🎮 Federated Learning Interactive Game
<div style="display: flex; gap: 10px;">
    <img src="/assets/non_iid_demo.gif" width="49%" alt="Non-IID FL Game Demo"/>
    <img src="/assets/iid_demo.gif" width="49%" alt="IID FL Game Demo"/>
</div>


Checkout the game at: [FL-Interactive-Game](https://amanpriyanshu.github.io/FL-Interactive-Game/)

An interactive web-based demonstration that makes learning about Federated Learning fun! Train models across multiple clients, experiment with different parameters, and watch your global model improve - all without sharing raw data.

## 🌟 Features

- **Interactive Client Management**: Control 5 independent clients training on MNIST data
- **Real-time Visualization**: Watch training progress with live accuracy plots
- **Configurable Parameters**: 
  - Per-client learning rates and epoch counts
  - Communication dropout probability
  - Data distribution settings (IID vs Non-IID)
- **Model Aggregation**: Combine client models into a stronger global model
- **Responsive Design**: Works seamlessly on both desktop and mobile devices

## 🎯 Learning Objectives

This game helps you understand:
- How Federated Learning works in practice
- Impact of different learning rates and training durations
- Effects of communication dropouts in distributed learning
- Differences between IID and Non-IID data distributions
- Model aggregation and its effects on global performance

## 🎲 How to Play

1. Choose your data distribution (IID or Non-IID)
   - IID: Balanced data
   - Non-IID: Chaos mode
   - *Remember to refresh when switching - those sneaky old models like to hang around! 🔄*

2. Configure each client:
   - Adjust learning rates (0.00001 to 0.001)
   - Set training epochs (5 to 15)

3. Set global parameters:
   - Adjust communication dropout probability
   - This simulates real-world network conditions

4. Start training:
   - Train clients individually or all at once
   - Watch the accuracy plots evolve
   - Aggregate models to improve global performance

## 🔧 Technical Details

- Built with TensorFlow.js
- Uses MNIST dataset (preprocessed for FL setting)
- Neural Network Architecture:
  - Input Layer: 784 neurons (28x28 images)
  - Hidden Layer: 64 neurons, ReLU activation
  - Output Layer: 10 neurons, Softmax activation

## 🏆 Challenge Goals

1. Beat 90% accuracy on the global model
2. Achieve consistent performance across all clients
3. Maintain good performance with high dropout rates
4. Master both IID and Non-IID scenarios

## Some Known Issues

- Model states persist between data distribution switches (refresh required)
- Some mobile devices may experience performance lag with multiple clients
- High learning rates can cause training instability
