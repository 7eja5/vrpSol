import random
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Depot:
    location: Point
    vehicles: int


def calculate_distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


def generate_test_data(num_depots: int, vehicles_per_depot: int,
                       num_customers: int, grid_size: int = 100) -> Tuple[List[Depot], List[Point]]:
    """Generate random test data for MDRP."""
    depots = []
    for _ in range(num_depots):
        depot = Depot(
            location=Point(
                random.uniform(0, grid_size),
                random.uniform(0, grid_size)
            ),
            vehicles=vehicles_per_depot
        )
        depots.append(depot)

    customers = [
        Point(
            random.uniform(0, grid_size),
            random.uniform(0, grid_size)
        ) for _ in range(num_customers)
    ]

    return depots, customers


def nearest_neighbor_mdrp(depots: List[Depot], customers: List[Point]) -> Tuple[List[List[List[Point]]], float]:
    """
    Solve MDRP using nearest neighbor heuristic.
    Returns routes for each vehicle and total distance.
    """
    unassigned_customers = customers.copy()
    routes = []
    total_distance = 0

    # For each depot
    for depot in depots:
        depot_routes = []

        # For each vehicle in the depot
        for _ in range(depot.vehicles):
            if not unassigned_customers:
                break

            route = [depot.location]
            current_point = depot.location
            route_distance = 0

            # While there are still customers to visit
            while unassigned_customers:
                # Find nearest customer
                nearest_customer = min(
                    unassigned_customers,
                    key=lambda c: calculate_distance(current_point, c)
                )

                # Add customer to route
                route.append(nearest_customer)
                route_distance += calculate_distance(current_point, nearest_customer)
                current_point = nearest_customer
                unassigned_customers.remove(nearest_customer)

                if not unassigned_customers:
                    break

            # Return to depot
            route.append(depot.location)
            route_distance += calculate_distance(current_point, depot.location)
            total_distance += route_distance
            depot_routes.append(route)

        routes.append(depot_routes)

    return routes, total_distance


def print_solution(routes: List[List[List[Point]]], total_distance: float):
    """Print the solution in a readable format."""
    print(f"\nTotal distance: {total_distance:.2f}")
    for depot_idx, depot_routes in enumerate(routes):
        print(f"\nDepot {depot_idx + 1}:")
        for vehicle_idx, route in enumerate(depot_routes):
            print(f"Vehicle {vehicle_idx + 1} route:")
            for point in route:
                print(f"({point.x:.2f}, {point.y:.2f})", end=" -> ")
            print("END")


def main():
    # Test case 1: Small instance
    print("\nTest Case 1: Small Instance")
    depots, customers = generate_test_data(
        num_depots=2,
        vehicles_per_depot=2,
        num_customers=8
    )
    routes, total_distance = nearest_neighbor_mdrp(depots, customers)
    print_solution(routes, total_distance)

    # Test case 2: Medium instance
    print("\nTest Case 2: Medium Instance")
    depots, customers = generate_test_data(
        num_depots=3,
        vehicles_per_depot=3,
        num_customers=15
    )
    routes, total_distance = nearest_neighbor_mdrp(depots, customers)
    print_solution(routes, total_distance)


if __name__ == "__main__":
    main()