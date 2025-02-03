import math
import numpy as np

# CONFIGURABLE PARAMETERS
# ==============================================

# ball properties
BALL_DIAMETER = 49.2 / 1000  # meters
BALL_MASS = 27 / 1000    # kg
INITIAL_VELOCITY = BALL_DIAMETER/0.0047  # m/s

# environment properties
AIR_DENSITY = 1.225        # ρ (rho): density of air (kg/m³)
GRAVITY = 9.8              # g: acceleration due to gravity (m/s²)
DRAG_COEFFICIENT = 0.55    # Cd: drag coefficient for a sphere

# barrel properties
BARREL_LENGTH = 0.66       # meters - affects initial x position
BARREL_HEIGHT = 0.06       # meters - base height
BARREL_PIVOT_POINT = 0.3   # meters - distance from pivot to end of barrel

# simulation
TIME_STEP = 0.001          # dt: time step for numerical integration
TARGET_TOLERANCE = 0       # meters

def simulate_trajectory(theta, target_x, target_y, show_path=False):
    """Simulates projectile motion with air resistance using numerical integration"""
    
    # Calculate ball's cross-sectional area
    # A = πr² where r is radius
    R = BALL_DIAMETER / 2
    A = math.pi * R ** 2  # Cross-sectional area for drag calculation
    
    # Gravity vector g = [0, -9.8] m/s²
    g = np.array([0, -GRAVITY])
    
    # Initial position calculation based on barrel geometry
    # x = -L*cos(θ) where L is barrel length
    # y = h + d*sin(θ) where h is base height, d is pivot distance
    initial_x = -(BARREL_LENGTH * math.cos(theta))
    initial_y = BARREL_HEIGHT + BARREL_PIVOT_POINT * math.sin(theta)
    
    pos = np.array([initial_x, initial_y])
    prev_pos = pos.copy()
    
    # Initial momentum p = mv
    # v = v₀[cos(θ), sin(θ)] where v₀ is initial speed
    v_initial = np.array([math.cos(theta), math.sin(theta)]) * INITIAL_VELOCITY
    p = BALL_MASS * v_initial
    
    t = 0
    dt = TIME_STEP
    min_dist_to_target = float('inf')
    hit_target = False
    hit_position = None
    
    # Main physics loop - continues until ball hits ground (y=0)
    while pos[1] >= 0:
        # Calculate distance to target using Pythagorean theorem
        dist_to_target = np.linalg.norm([pos[0] - target_x, pos[1] - target_y])
        
        # Check for target hit by seeing if we crossed target_x at right height
        if (abs(pos[1] - target_y) < TARGET_TOLERANCE and 
            ((prev_pos[0] <= target_x and pos[0] >= target_x) or 
             (prev_pos[0] >= target_x and pos[0] <= target_x))):
            hit_target = True
            hit_position = pos.copy()
            if not show_path:
                break
            
        min_dist_to_target = min(min_dist_to_target, dist_to_target)
        prev_pos = pos.copy()
        
        # Current velocity v = p/m
        v = p / BALL_MASS
        
        # Calculate velocity magnitude |v| = √(v_x² + v_y²)
        v_mag = np.linalg.norm(v)
        
        # Unit vector in velocity direction v̂ = v/|v|
        v_norm = v / v_mag if v_mag > 0 else v
        
        # Force calculations:
        # 1. Gravity: F_g = mg
        # 2. Air resistance: F_d = -½ρAv²Cd * v̂
        # Total force: F = F_g + F_d
        F = (BALL_MASS * g - 
             0.5 * AIR_DENSITY * A * DRAG_COEFFICIENT * v_mag ** 2 * v_norm)
        
        # Update momentum using F = dp/dt → p = p + F*dt
        p = p + F * dt
        
        # Update position using v = dx/dt → x = x + v*dt
        # Where v = p/m
        pos = pos + p * dt / BALL_MASS
        
        t += dt
    
    if show_path:
        print(f"Angle: {math.degrees(theta):.1f}°")
        print(f"Time: {t:.3f}s")
        if hit_position is not None:
            print(f"Hit target at: ({hit_position[0]:.3f}m, {hit_position[1]:.3f}m)")
        print(f"Landing: ({pos[0]:.3f}m, 0m)\n")
        
    return 0 if hit_target else min_dist_to_target

def find_launch_angles(target_x, target_y):
    """
    Finds optimal launch angles using a sweep method:
    1. Tests angles from 5° to 85° in 0.1° increments
    2. Finds angles that result in smallest distance to target
    3. Returns up to 2 angles that are >5° apart
    """
    angles = []
    distances = []
    
    # Test angles from 5° to 85° with 0.1° steps
    test_angles = np.linspace(5, 85, 801)
    
    for angle in test_angles:
        theta = math.radians(angle)
        distance = simulate_trajectory(theta, target_x, target_y)
        angles.append(theta)
        distances.append(distance)
    
    distances = np.array(distances)
    sorted_indices = np.argsort(distances)
    
    best_angles = []
    for idx in sorted_indices:
        angle = math.degrees(angles[idx])
        angle = round(angle * 10) / 10
        
        if not best_angles:
            best_angles.append(angle)
        elif len(best_angles) == 1:
            if abs(angle - best_angles[0]) > 5:
                best_angles.append(angle)
                break
    
    return [math.radians(angle) for angle in sorted(best_angles)]

def main():
    target_x = float(input("Enter target distance (in meters): "))
    target_y = float(input("Enter target height (in meters): "))
    
    angles = find_launch_angles(target_x, target_y)
    
    if len(angles) == 0:
        print(f"No solution found - target may be out of range")
    else:
        print(f"\nBest launch angles to reach target at ({target_x}m, {target_y}m):")
        for angle in angles:
            print(f"  {math.degrees(angle):.1f}°")
            simulate_trajectory(angle, target_x, target_y, show_path=True)

if __name__ == "__main__":
    main()