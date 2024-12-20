import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class MDVRPSolver:
    def __init__(self):
        # Sample data generation
        self.num_depots = 2
        self.num_vehicles_per_depot = 2
        self.num_customers = 12

        # Generate random coordinates for depots and customers
        np.random.seed(42)  # For reproducibility
        self.depot_coords = np.random.rand(self.num_depots, 2) * 100
        self.customer_coords = np.random.rand(self.num_customers, 2) * 100

        # Vehicle capacity and customer demands
        self.vehicle_capacity = 100
        self.customer_demands = np.random.randint(10, 30, size=self.num_customers)

    def calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def find_nearest_unvisited(self, current_pos: np.ndarray,
                               unvisited: List[int]) -> Tuple[int, float]:
        """Find the nearest unvisited customer."""
        min_dist = float('inf')
        nearest_idx = -1

        for cust_idx in unvisited:
            dist = self.calculate_distance(current_pos,
                                           self.customer_coords[cust_idx])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = cust_idx

        return nearest_idx, min_dist

    def solve(self) -> Dict:
        """Solve the MDVRP using nearest neighbor heuristic."""
        unvisited = list(range(self.num_customers))
        routes = {i: [] for i in range(self.num_depots)}
        total_distance = 0

        # Assign customers to nearest depot first
        depot_customers = {i: [] for i in range(self.num_depots)}

        for cust_idx in range(self.num_customers):
            min_dist = float('inf')
            assigned_depot = -1

            for depot_idx in range(self.num_depots):
                dist = self.calculate_distance(self.depot_coords[depot_idx],
                                               self.customer_coords[cust_idx])
                if dist < min_dist:
                    min_dist = dist
                    assigned_depot = depot_idx

            depot_customers[assigned_depot].append(cust_idx)

        # Create routes for each depot
        for depot_idx in range(self.num_depots):
            remaining = depot_customers[depot_idx].copy()

            for vehicle in range(self.num_vehicles_per_depot):
                if not remaining:
                    break

                current_route = []
                current_pos = self.depot_coords[depot_idx]
                current_capacity = self.vehicle_capacity

                while remaining:
                    nearest, dist = self.find_nearest_unvisited(current_pos,
                                                                remaining)

                    if nearest == -1:
                        break

                    if (current_capacity -
                            self.customer_demands[nearest] >= 0):
                        current_route.append(nearest)
                        current_capacity -= self.customer_demands[nearest]
                        current_pos = self.customer_coords[nearest]
                        remaining.remove(nearest)
                        total_distance += dist
                    else:
                        break

                if current_route:
                    # Add return to depot distance
                    total_distance += self.calculate_distance(
                        current_pos, self.depot_coords[depot_idx])
                    routes[depot_idx].append(current_route)

        return {"routes": routes, "total_distance": total_distance}

    def visualize_solution(self, routes: Dict):
        """Visualize the solution using matplotlib."""
        plt.figure(figsize=(10, 10))

        # Plot depots
        plt.scatter(self.depot_coords[:, 0], self.depot_coords[:, 1],
                    c='red', s=100, label='Depots')

        # Plot customers
        plt.scatter(self.customer_coords[:, 0], self.customer_coords[:, 1],
                    c='blue', s=50, label='Customers')

        # Plot routes
        colors = ['g', 'm', 'y', 'k']
        for depot_idx, depot_routes in routes.items():
            for route_idx, route in enumerate(depot_routes):
                color = colors[(depot_idx + route_idx) % len(colors)]

                # Draw line from depot to first customer
                if route:
                    plt.plot([self.depot_coords[depot_idx][0],
                              self.customer_coords[route[0]][0]],
                             [self.depot_coords[depot_idx][1],
                              self.customer_coords[route[0]][1]],
                             c=color)

                # Draw lines between customers
                for i in range(len(route) - 1):
                    plt.plot([self.customer_coords[route[i]][0],
                              self.customer_coords[route[i + 1]][0]],
                             [self.customer_coords[route[i]][1],
                              self.customer_coords[route[i + 1]][1]],
                             c=color)

                # Draw line from last customer back to depot
                if route:
                    plt.plot([self.customer_coords[route[-1]][0],
                              self.depot_coords[depot_idx][0]],
                             [self.customer_coords[route[-1]][1],
                              self.depot_coords[depot_idx][1]],
                             c=color)

        plt.legend()
        plt.title('Multi-Depot VRP Solution')
        plt.show()


# Run the solver
solver = MDVRPSolver()
solution = solver.solve()
print(f"Total distance: {solution['total_distance']:.2f}")
print("\nRoutes:")
for depot_idx, routes in solution['routes'].items():
    print(f"Depot {depot_idx}:")
    for route_idx, route in enumerate(routes):
        print(f"  Vehicle {route_idx}: {route}")

solver.visualize_solution(solution['routes'])