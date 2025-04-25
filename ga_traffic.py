import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load or create model for signal optimization
try:
    signal_model = joblib.load('signal_model.pkl')
except:
    signal_model = RandomForestRegressor(n_estimators=50)

# Genetic Algorithm with machine learning enhancement
def optimize_signal_time(traffic_density, further_traffic):
    # Try ML prediction first
    try:
        ml_prediction = signal_model.predict([[traffic_density, further_traffic]])[0]
        ml_prediction = max(10, min(120, ml_prediction))
    except:
        ml_prediction = 30  # Default fallback
    
    # Genetic Algorithm parameters
    population_size = 20
    num_generations = 50
    mutation_rate = 0.1
    
    # Initialize population around ML prediction
    population = np.clip(
        np.random.normal(ml_prediction, 15, population_size),
        10, 120
    ).astype(int)
    
    # Fitness function weights
    weights = {
        'throughput': 0.6,
        'wait_time': -0.3,
        'emergency': 1.0,
        'pedestrian': 0.5
    }
    
    # Genetic Algorithm optimization
    for _ in range(num_generations):
        # Evaluate fitness
        fitness = []
        for signal_time in population:
            # Simulate outcomes (simplified for demo)
            throughput = min(traffic_density, signal_time * 0.5)
            wait_time = max(0, traffic_density - throughput)
            
            # Calculate fitness
            score = (
                weights['throughput'] * throughput +
                weights['wait_time'] * wait_time
            )
            fitness.append(score)
        
        # Selection (tournament) - FIXED THIS SECTION
        new_population = []
        for _ in range(population_size):
            # Tournament selection
            candidates = np.random.choice(population, size=3)
            winner = candidates[np.argmax([fitness[np.where(population == c)[0][0]] for c in candidates])]
            new_population.append(winner)
        
        # Crossover and mutation
        population = []
        for i in range(0, len(new_population), 2):
            parent1 = new_population[i]
            parent2 = new_population[i+1] if i+1 < len(new_population) else new_population[0]
            
            # Blend crossover
            alpha = np.random.random()
            child = alpha * parent1 + (1-alpha) * parent2
            
            # Mutation
            if np.random.random() < mutation_rate:
                child += np.random.randint(-10, 10)
            
            population.append(int(np.clip(child, 10, 120)))
    
    # Return best solution
    best_index = np.argmax(fitness)
    return int(population[best_index])

# Function to update model with new data
def update_signal_model(X, y):
    global signal_model
    try:
        signal_model.fit(X, y)
        joblib.dump(signal_model, 'signal_model.pkl')
    except Exception as e:
        print(f"Failed to update signal model: {e}")