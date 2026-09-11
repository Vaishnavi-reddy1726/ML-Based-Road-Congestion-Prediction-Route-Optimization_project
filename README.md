# ML-Based Road Congestion Prediction & Route Optimization

An end-to-end machine learning system that predicts road congestion using weather and temporal patterns, then uses those predictions to optimize routes through a weighted road-network graph.

The system combines **machine learning, graph algorithms, and shortest-path optimization** to identify the fastest route between two points under different traffic and weather conditions.

> **Note on Data:** No real-world dataset is included with this project. A synthetic data generator (`src/generate_data.py`) creates a realistic road network and 60 days of hourly weather and traffic data, including rush-hour spikes, weekend effects, and weather-related slowdowns.
>
> Real datasets can be used by replacing `data/road_network.csv` and `data/historical_data.csv` while maintaining the same schema.

## Project Structure

```text
road_congestion_project/
│
├── data/
│   ├── road_network.csv
│   ├── historical_data.csv
│   └── cleaned_data.csv
│
├── models/
│   └── congestion_model.joblib
│
├── outputs/
│   ├── feature_importance.png
│   └── predicted_vs_actual.png
│
├── src/
│   ├── generate_data.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── graph_network.py
│   ├── dijkstra.py
│   ├── route_optimizer.py
│   └── main.py
│
├── requirements.txt
└── README.md

How it works

### **1. Data Generation & Preprocessing**

### **2. Feature Engineering**

### **3. Congestion Prediction**

### **4. Dynamic Road-Network Graph**

### **5. Route Optimization**

## **Running the Project**

## **Model Evaluation**

## **Dataset Schema**

## **Future Improvements**

## **Technologies Used**
