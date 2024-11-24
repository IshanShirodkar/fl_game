# 🎮 Federated Learning Interactive Game

An interactive web-based demonstration that makes learning about Federated Learning fun! Train models across multiple clients, experiment with different parameters, and watch your global model improve - all without sharing raw data.

![FL Game Demo](demo.gif)

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

## 🚀 Getting Started

Visit the webpage at: [FL-Interactive-Game]()

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MNIST Dataset providers
- TensorFlow.js team

## 🐛 Known Issues

- Model states persist between data distribution switches (refresh required)
- Some mobile devices may experience performance lag with multiple clients
- High learning rates can cause training instability
