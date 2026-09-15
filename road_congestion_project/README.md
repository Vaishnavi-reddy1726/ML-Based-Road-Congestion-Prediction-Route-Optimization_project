# ML-Based Road Congestion Prediction & Route Optimization

An end-to-end system that predicts road-traffic congestion from weather and
temporal patterns using machine learning, then feeds those predictions into
a weighted road-network graph to compute the fastest route between two
points using Dijkstra's shortest-path algorithm.

**Verified results on the included synthetic network** (552 road segments,
130 intersections, 60 days of hourly weather + traffic history):
- Regression model (continuous 0-1 congestion score): **R² = 0.962**, MAE = 0.038
- 3-level classifier (Low / Medium / High congestion): **90.13% held-out accuracy**
- 8 of 17 engineered features carry ~99.8% of the predictive signal (see `outputs/feature_importance.png`)

> **Note on data:** No dataset was provided, so this project includes a
> synthetic data generator (`src/generate_data.py`) that produces a
> realistic road network and 60 days of hourly weather + traffic history
> (rush-hour spikes, weekend effects, and weather-driven slowdowns baked in).
> The numbers above are genuine outputs of this synthetic data — swap in
> real data (same schema, see below) and these figures will change.

## Project structure

```
road_congestion_project/
├── data/
│   ├── road_network.csv        # generated: road segments (graph edges)
│   ├── historical_data.csv     # generated: hourly weather + traffic history
│   └── cleaned_data.csv        # generated: cleaned dataset
├── models/
│   └── congestion_model.joblib # generated: trained ML model
├── outputs/
│   ├── feature_importance.png  # generated: model evaluation plot
│   └── predicted_vs_actual.png # generated: model evaluation plot
├── src/
│   ├── generate_data.py        # synthetic road network + historical data
│   ├── data_preprocessing.py   # cleaning: missing values, outliers, dtypes
│   ├── feature_engineering.py  # time/weather features, rolling averages
│   ├── model_training.py       # trains + saves the regression model
│   ├── classification_model.py # trains + saves the 3-level classifier
│   ├── model_evaluation.py     # metrics + evaluation plots
│   ├── graph_network.py        # weighted road-network graph
│   ├── dijkstra.py             # Dijkstra's algorithm (heapq priority queue)
│   ├── route_optimizer.py      # ties ML predictions + graph + Dijkstra together
│   └── main.py                 # runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## How it works

1. **Data pipeline** (`generate_data.py` → `data_preprocessing.py` →
   `feature_engineering.py`): raw hourly records of weather (temperature,
   precipitation, visibility, wind, condition) and road congestion are
   cleaned (missing values, outlier removal, dedup) and turned into model
   features — cyclical hour/day encodings, rush-hour and weekend flags, a
   weather-severity index, one-hot weather condition, and a per-road
   rolling-average congestion lag feature.

2. **Model training** (`model_training.py`, `classification_model.py`): a
   `RandomForestRegressor` predicts a continuous congestion level (0 = free-flowing,
   1 = gridlock) for live route optimization, using a **chronological** train/test
   split (no shuffling) since this is time-ordered data. A separate
   `RandomForestClassifier` buckets the same signal into three operational
   levels (Low / Medium / High) for a single interpretable accuracy metric.
   On the included 552-segment synthetic network: regression R² = 0.962,
   classifier accuracy = 90.13%.

3. **Graph construction** (`graph_network.py`): the road network is loaded
   into a weighted directed graph where each edge (road segment) has a
   distance and a free-flow speed limit. Edge weights (travel time) are
   dynamically recomputed from ML-predicted congestion:

   ```
   effective_speed = base_speed_kmph * (1 - congestion_level * 0.85)
   travel_time_min = (distance_km / effective_speed) * 60
   ```

4. **Route optimization** (`route_optimizer.py` + `dijkstra.py`): for a
   given origin, destination, time, and weather scenario, the model
   predicts congestion for every road segment, the graph weights are
   refreshed, and Dijkstra's algorithm (implemented with a `heapq` binary
   heap priority queue) computes the minimum-travel-time path.

## Running it

```bash
pip install -r requirements.txt
cd src
python main.py
```

> The `data/road_network.csv` for the verified 552-segment network is
> included, but `historical_data.csv`, `cleaned_data.csv`, and the trained
> `.joblib` models are **not** shipped in this zip (they're 40-100MB+ and
> fully reproducible). Running `main.py` regenerates all of it deterministically
> (fixed random seed) in under a minute.

This will:
- generate the synthetic dataset (only if `data/` is empty),
- clean it and engineer features,
- train and evaluate the congestion model,
- save the trained model to `models/`,
- run a sample route-optimization query (rainy Wednesday 8 AM rush hour)
  and print the fastest route with per-segment congestion and travel time.

To regenerate evaluation plots separately:

```bash
python model_evaluation.py
```

## Using it with real data

Replace the generated CSVs with real data using this schema:

**`data/road_network.csv`**
| column | description |
|---|---|
| road_id | unique segment id |
| from_node | origin intersection/node id |
| to_node | destination intersection/node id |
| distance_km | segment length |
| base_speed_kmph | free-flow speed limit |

**`data/historical_data.csv`**
| column | description |
|---|---|
| road_id | segment id (matches road_network.csv) |
| timestamp | datetime of observation |
| hour, day_of_week, is_weekend | time fields |
| temperature_c, precipitation_mm, visibility_km, wind_speed_kmph | weather |
| weather_condition | categorical (Clear/Rain/Fog/Storm/Snow, etc.) |
| congestion_level | observed congestion, 0–1 (or derived from speed/volume) |

Then just re-run `main.py` — everything downstream (cleaning, features,
model, graph, routing) adapts automatically.

## Extending

- Swap `RandomForestRegressor` for `GradientBoostingRegressor` or an
  XGBoost/LightGBM model in `model_training.py`.
- Add live weather API integration in `route_optimizer.py` in place of the
  manually specified `weather` dict.
- Add k-shortest-paths (Yen's algorithm) for alternate route suggestions.
- Swap the custom `dijkstra.py` for `networkx` if you'd rather use a
  battle-tested graph library for more complex network operations.
