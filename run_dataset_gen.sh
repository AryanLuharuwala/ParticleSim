#!/bin/bash
set -e

# Generate templates
echo "Generating rock templates..."
python3 generate_templates.py

# Run LAMMPS
echo "Running LAMMPS simulation..."
lmp -in simulation.in

echo "Done! Images should be generated."
