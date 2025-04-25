import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from typing import List, Tuple

class TrafficSignalOptimizer:
    """
    Advanced traffic signal optimization using hybrid genetic algorithm and machine learning
    """
    def __init__(self, num_signals: int = 4):
        self.num_signals = num_signals
        self.params = {
            'min_green': 10,
            'max_green': 60,
            'min_yellow': 3,
            'max_yellow': 10,
            'min_red': 10,
            'max_red': 60,
            'population_size': 100,
            'generations': 200,
            'mutation_rate': 0.15
        }
        self.model = self._load_model()
        
    def _load_model(self):
        """Load or create ML model for fitness prediction"""
        try:
            return joblib.load('traffic_ga_model.pkl')
        except:
            return RandomForestRegressor(n_estimators=50)
    
    def evaluate_fitness(self, 
                       signal_times: List[Tuple[int, int, int]], 
                       vehicle_counts: List[int],
                       emergency: bool = False,
                       pedestrian: bool = False) -> float:
        """
        Enhanced fitness evaluation with multiple factors
        """
        # ML-based prediction component
        ml_input = np.array([
            [g, y, r, v, int(emergency), int(pedestrian)]
            for (g, y, r), v in zip(signal_times, vehicle_counts)
        ])
        
        try:
            ml_score = self.model.predict(ml_input).mean()
        except:
            ml_score = 0
        
        # Traditional metrics
        throughput = sum(
            min(v, g * 0.8 + y * 0.2)  # Estimated vehicles cleared
            for (g, y, r), v in zip(signal_times, vehicle_counts)
        )
        
        avg_wait = sum(
            max(0, v - (g * 0.8 + y * 0.2)) / self.num_signals  # Estimated wait
        )
        
        # Special conditions
        emergency_penalty = 1000 if emergency and any(r > 15 for _, _, r in signal_times) else 0
        pedestrian_bonus = 500 if pedestrian and any(g >= 20 for g, _, _ in signal_times) else 0
        
        # Composite fitness
        return (
            0.7 * throughput 
            - 0.4 * avg_wait 
            + 0.3 * ml_score
            - emergency_penalty
            + pedestrian_bonus
        )
    
    def create_individual(self) -> List[Tuple[int, int, int]]:
        """Create random signal timing individual"""
        return [
            (
                random.randint(self.params['min_green'], self.params['max_green']),
                random.randint(self.params['min_yellow'], self.params['max_yellow']),
                random.randint(self.params['min_red'], self.params['max_red'])
            ) for _ in range(self.num_signals)
        ]
    
    def crossover(self, 
                 parent1: List[Tuple[int, int, int]], 
                 parent2: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Enhanced crossover with signal-wise blending"""
        return [
            (
                (g1 + g2) // 2 if random.random() < 0.7 else (g1 if random.random() < 0.5 else g2),
                (y1 + y2) // 2 if random.random() < 0.7 else (y1 if random.random() < 0.5 else y2),
                (r1 + r2) // 2 if random.random() < 0.7 else (r1 if random.random() < 0.5 else r2)
            ) for (g1, y1, r1), (g2, y2, r2) in zip(parent1, parent2)
        ]
    
    def mutate(self, individual: List[Tuple[int, int, int]]]) -> List[Tuple[int, int, int]]:
        """Smart mutation that considers traffic patterns"""
        return [
            (
                max(self.params['min_green'], min(self.params['max_green'], 
                    g + random.randint(-5, 5) if random.random() < self.params['mutation_rate'] else g)),
                max(self.params['min_yellow'], min(self.params['max_yellow'], 
                    y + random.randint(-2, 2) if random.random() < self.params['mutation_rate'] else y)),
                max(self.params['min_red'], min(self.params['max_red'], 
                    r + random.randint(-5, 5) if random.random() < self.params['mutation_rate'] else r))
            ) for g, y, r in individual
        ]
    
    def optimize(self, 
                vehicle_counts: List[int],
                emergency: bool = False,
                pedestrian: bool = False) -> Tuple[List[Tuple[int, int, int]], float]:
        """
        Run full optimization cycle
        Returns best signal timings and fitness score
        """
        population = [self.create_individual() for _ in range(self.params['population_size'])]
        best_individual = None
        best_fitness = -float('inf')
        
        for generation in range(self.params['generations']):
            # Evaluate fitness
            fitness_scores = [
                self.evaluate_fitness(ind, vehicle_counts, emergency, pedestrian)
                for ind in population
            ]
            
            # Track best solution
            current_best = max(fitness_scores)
            if current_best > best_fitness:
                best_fitness = current_best
                best_individual = population[fitness_scores.index(current_best)]
            
            # Selection (tournament)
            parents = []
            for _ in range(self.params['population_size']):
                tournament = random.sample(
                    range(self.params['population_size']), 
                    min(5, self.params['population_size']))
                winner = tournament[0]
                for i in tournament[1:]:
                    if fitness_scores[i] > fitness_scores[winner]:
                        winner = i
                parents.append(population[winner])
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(parents), 2):
                if i+1 < len(parents):
                    child = self.crossover(parents[i], parents[i+1])
                else:
                    child = parents[i]
                new_population.append(self.mutate(child))
            
            population = new_population
        
        return best_individual, best_fitness
    
    def train_model(self, historical_data: List[dict]):
        """
        Train the ML model with historical data
        Data format: [
            {
                'signal_times': [(g,y,r), ...],
                'vehicle_counts': [int, ...],
                'emergency': bool,
                'pedestrian': bool,
                'throughput': float,
                'wait_time': float
            },
            ...
        ]
        """
        try:
            X = []
            y = []
            for entry in historical_data:
                for (g, y, r), v in zip(entry['signal_times'], entry['vehicle_counts']):
                    X.append([g, y, r, v, int(entry['emergency']), int(entry['pedestrian'])])
                    y.append(entry['throughput'] - 0.5 * entry['wait_time'])
            
            self.model.fit(X, y)
            joblib.dump(self.model, 'traffic_ga_model.pkl')
            return True
        except Exception as e:
            print(f"Training failed: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize optimizer for 4 intersections
    optimizer = TrafficSignalOptimizer(num_signals=4)
    
    # Example vehicle counts (per intersection)
    vehicle_counts = [45, 30, 60, 25]
    
    # Run optimization
    best_timings, fitness = optimizer.optimize(
        vehicle_counts,
        emergency=False,
        pedestrian=True
    )
    
    print("Optimized Signal Timings (G,Y,R):")
    for i, (g, y, r) in enumerate(best_timings):
        print(f"Intersection {i+1}: {g}s, {y}s, {r}s")
    print(f"Fitness Score: {fitness:.2f}")
    
    # To train the model (would need actual historical data):
    # optimizer.train_model(historical_data)