import bpy
import random
import math
import os
import sys

# Add current dir to path if needed for local modules
sys.path.append(os.getcwd())

class MiningConveyorSimulator:
    def __init__(self, output_dir="dataset", num_images=10):
        self.output_dir = output_dir
        self.num_images = num_images
        self.scene = bpy.context.scene
        self.collection = bpy.data.collections.get("Collection")
        if not self.collection:
            self.collection = bpy.data.collections.new("Collection")
            bpy.context.scene.collection.children.link(self.collection)
            
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.setup_render_settings()
        self.setup_compositor()
        
    def clear_scene(self):
        # Delete everything
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Remove orphan data blocks to prevent memory leaks
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)
        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)
        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)

    def setup_render_settings(self):
        # Use Cycles for realism
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.device = 'GPU'
        self.scene.cycles.samples = 64
        # Denoising
        self.scene.cycles.use_denoising = True
        
        self.scene.render.resolution_x = 512
        self.scene.render.resolution_y = 512
        self.scene.render.resolution_percentage = 100
        
    def setup_compositor(self):
        self.scene.use_nodes = True
        tree = self.scene.node_tree
        
        # Clear default nodes
        for node in tree.nodes:
            tree.nodes.remove(node)
            
        # Create input
        rl_layer = tree.nodes.new('CompositorNodeRLayers')
        rl_layer.location = (0, 0)
        
        # Enable Object Index pass for segmentation
        self.scene.view_layers["ViewLayer"].use_pass_object_index = True
        
        # Output RGB
        file_output_node = tree.nodes.new('CompositorNodeOutputFile')
        file_output_node.base_path = self.output_dir
        file_output_node.location = (400, 0)
        
        # Configure file slots
        # Default node usually has one slot named 'Image'. We reuse it for 'rgb'.
        slots = file_output_node.file_slots
        if len(slots) > 0:
            slots[0].path = 'rgb'
        else:
            slots.new('rgb')
            
        # Link RGB to first input
        tree.links.new(rl_layer.outputs['Image'], file_output_node.inputs[0])
        
        # Create second slot for Mask
        slots.new('mask')
        
        # Link IndexOB to second input
        # Note: IndexOB is a value, but saving as image might normalize it or save as float.
        # Saved as PNG, it might be black for low indices (1, 2, 3...) if not normalized.
        # But for valid data, EXR is better, or we trust user can normalize. 
        # For visualization, let's keep it direct for now.
        tree.links.new(rl_layer.outputs['IndexOB'], file_output_node.inputs[1])

    def create_material(self, name):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        
        # Random dark grey/brown rock color
        base_color = (
            random.uniform(0.1, 0.4), 
            random.uniform(0.1, 0.4), 
            random.uniform(0.1, 0.4), 
            1.0
        )
        bsdf.inputs['Base Color'].default_value = base_color
        bsdf.inputs['Roughness'].default_value = random.uniform(0.6, 0.9)
        
        # Add some noise for bump
        tex_noise = nodes.new("ShaderNodeTexNoise")
        tex_noise.inputs['Scale'].default_value = 50.0
        
        bump = nodes.new("ShaderNodeBump")
        bump.inputs['Strength'].default_value = 0.2
        
        mat.node_tree.links.new(tex_noise.outputs['Fac'], bump.inputs['Height'])
        mat.node_tree.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
        
        return mat

    def create_rock(self, location, index):
        # Create a base cube or icosphere
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, location=location)
        obj = bpy.context.active_object
        obj.name = f"Rock_{index}"
        
        # Ensure Object Index is set for segmentation (1-indexed)
        obj.pass_index = index + 1
        
        # Add modifiers for shape variation
        
        # 1. Subdivision to ensure enough geometry for displacement
        subsurf = obj.modifiers.new(name="Subsurf", type='SUBSURF')
        subsurf.levels = 2
        
        # 2. Displace with Voronoi or Cloud texture
        disp = obj.modifiers.new(name="Displace", type='DISPLACE')
        tex = bpy.data.textures.new(f"Texture_{index}", 'CLOUDS')
        tex.noise_scale = random.uniform(0.5, 1.5)
        disp.texture = tex
        disp.strength = random.uniform(0.05, 0.2)
        
        # 3. Random scale (non-uniform)
        s = random.uniform(0.3, 0.8) # Base size
        obj.scale = (
            s * random.uniform(0.8, 1.2),
            s * random.uniform(0.8, 1.2),
            s * random.uniform(0.8, 1.2)
        )
        
        # material
        mat = self.create_material(f"Mat_{index}")
        obj.data.materials.append(mat)
        
        # Physics - Active Rigid Body
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = 'ACTIVE'
        obj.rigid_body.mass = 1.0
        obj.rigid_body.friction = 0.8
        obj.rigid_body.collision_shape = 'CONVEX_HULL'
        
        return obj

    def setup_scene(self):
        # Conveyor Belt (Static Plane)
        bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
        belt = bpy.context.active_object
        belt.name = "ConveyorBelt"
        
        # Dark grey belt material
        mat = bpy.data.materials.new(name="BeltMat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1.0)
        belt.data.materials.append(mat)
        
        # Physics - Passive
        bpy.ops.rigidbody.object_add()
        belt.rigid_body.type = 'PASSIVE'
        belt.rigid_body.friction = 0.8
        
        # Walls to keep rocks in view
        bpy.ops.mesh.primitive_cube_add(scale=(1, 5, 1), location=(-2.5, 0, 1))
        wall_l = bpy.context.active_object
        bpy.ops.rigidbody.object_add()
        wall_l.rigid_body.type = 'PASSIVE'
        wall_l.hide_render = True
        
        bpy.ops.mesh.primitive_cube_add(scale=(1, 5, 1), location=(2.5, 0, 1))
        wall_r = bpy.context.active_object
        bpy.ops.rigidbody.object_add()
        wall_r.rigid_body.type = 'PASSIVE'
        wall_r.hide_render = True
        
        # Light
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        sun = bpy.context.active_object
        sun.data.energy = 5.0
        
        # Camera
        bpy.ops.object.camera_add(location=(0, 0, 7), rotation=(0, 0, 0))
        cam = bpy.context.active_object
        self.scene.camera = cam

    def run(self):
        print("Starting Dataset Generation...")
        
        for i in range(self.num_images):
            print(f"Generating Image {i+1}/{self.num_images}")
            self.clear_scene()
            self.setup_scene()
            
            # Spawn Rocks
            num_rocks = random.randint(15, 30)
            for r_idx in range(num_rocks):
                # Spawn high up with random xy spread
                loc = (
                    random.uniform(-1.5, 1.5),
                    random.uniform(-3, 3),
                    random.uniform(2, 5) # Height
                )
                self.create_rock(loc, r_idx)
                
            # Simulate Physics
            # Run for 50 frames to let rocks fall and settle
            start_frame = 1
            settle_frames = 60
            self.scene.frame_end = settle_frames
            
            # Bake/Run simulation by stepping frames?
            # In background mode, we just set the frame.
            # But we need to ensure physics detects the steps.
            # Usually setting current frame to N will compute up to N if cache is expected?
            # safer to loop
            for f in range(start_frame, settle_frames + 1):
                self.scene.frame_set(f)
                
            # Render
            self.scene.frame_set(settle_frames)
            
            # Update output filename based on frame/index
            # The File Output node appends frame number, so we set frame number to 'i'
            # But physics usually depends on frames 1..N.
            # So we might need to reset physics cache or similar?
            # Easier: Just render at frame 'settle_frames' but use the output node to name it properly.
            # Actually, File Output node uses the current frame number in filename.
            # We want unique filenames for each 'scene' generation.
            
            # Hack: We can manually rename files after render, or use the global frame counter
            # If we reuse the scene, we are resetting everything.
            
            # Let's clean up:
            # We are inside the loop. We generated rocks for this iteration.
            # We ran simulation to frame 60.
            # We render NOW.
            
            # Set output path prefix
            out_node = self.scene.node_tree.nodes.get("File Output")
            out_node.file_slots[0].path = f"rgb_{i:04d}_"
            out_node.file_slots[1].path = f"mask_{i:04d}_"
            
            # Render
            bpy.ops.render.render(write_still=False) # write_still=False because compositor writes files
            
        print("Dataset Generation Complete.")

if __name__ == "__main__":
    # Example usage: blender --background --python generate_dataset_blender.py
    gen = MiningConveyorSimulator(output_dir="/home/boom/startup/ParticleSim/dataset", num_images=5)
    gen.run()
