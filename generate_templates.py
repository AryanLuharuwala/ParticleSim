
import random
import os
import math

def generate_rock_template(rock_id, num_spheres, target_radius_cm):
    """
    Generates a LAMMPS molecule file for a clumped "rock".
    rock_id: ID of the rock (1, 2, ...)
    num_spheres: Number of constituent spheres (more = more irregular)
    target_radius_cm: Approximate radius of the heavy central part (defines overall size scale)
    """
    filename = f"rock{rock_id}.mol"
    
    spheres = []
    
    # 1. Main Center Sphere (The core of the rock)
    # Diameter is roughly 2x target radius
    d_main = target_radius_cm * 2.0 * random.uniform(0.8, 1.2)
    spheres.append({'x': 0.0, 'y': 0.0, 'z': 0.0, 'd': d_main})
    
    # 2. Add satellite spheres to create irregularity
    # They should be smaller and offset from center
    for _ in range(num_spheres - 1):
        # Scale satellite spheres relative to the main one
        d_sat = d_main * random.uniform(0.4, 0.9)
        r_sat = d_sat / 2.0
        r_main = d_main / 2.0
        
        # Offset distance: want them to overlap significantly but stick out
        # separation = (r_main + r_sat) * overlap_factor
        # overlap 0.5 means center of sat is on surface of main.
        # we want closer.
        offset_dist = r_main * random.uniform(0.4, 0.8) 
        
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        
        x = offset_dist * math.sin(phi) * math.cos(theta)
        y = offset_dist * math.sin(phi) * math.sin(theta)
        z = offset_dist * math.cos(phi)
        
        spheres.append({'x': x, 'y': y, 'z': z, 'd': d_sat})
        
    # Write to file
    with open(filename, 'w') as f:
        f.write(f"# Rock template {rock_id} (Approx Size: {target_radius_cm*2:.1f} cm)\n\n")
        f.write(f"{len(spheres)} atoms\n\n")
        
        f.write("Coords\n\n")
        for i, s in enumerate(spheres):
            f.write(f"{i+1} {s['x']:.4f} {s['y']:.4f} {s['z']:.4f}\n")
        f.write("\n")
        
        f.write("Types\n\n")
        for i in range(len(spheres)):
            f.write(f"{i+1} 1\n") 
        f.write("\n")
        
        f.write("Diameters\n\n")
        for i, s in enumerate(spheres):
            f.write(f"{i+1} {s['d']:.4f}\n")
        f.write("\n")
        
        f.write("Masses\n\n")
        # Density 2.5 g/cm^3
        density = 2.5 
        for i, s in enumerate(spheres):
            r = s['d'] / 2.0
            vol = (4.0/3.0) * math.pi * (r**3)
            mass = density * vol
            f.write(f"{i+1} {mass:.4f}\n")
        f.write("\n")

    print(f"Generated {filename}")

if __name__ == "__main__":
    random.seed(42)
    # Generate 10 variations
    # Sizes: 0.8 cm radius to 2.5 cm radius -> 1.6cm to 5.0cm diameter
    
    # 5 Small/Medium (1.5 - 2.5 cm diameter)
    for i in range(1, 6):
        target_radius = random.uniform(0.75, 1.25)
        num = random.randint(3, 5)
        generate_rock_template(i, num, target_radius)
        
    # 5 Large (3.0 - 5.0 cm diameter)
    for i in range(6, 11):
        target_radius = random.uniform(1.5, 2.5)
        num = random.randint(5, 9)
        generate_rock_template(i, num, target_radius)
