import bpy
import random
import math
import os
import sys
import json
import bpy_extras
from mathutils import Vector
# Add current dir to path if needed for local modules
sys.path.append(os.getcwd())

class MiningConveyorSimulator:
    def __init__(self, output_dir="dataset", num_images=10, use_raycast_visibility=False, use_exact_volume=False):
        self.output_dir = output_dir
        self.num_images = num_images
        self.use_raycast_visibility = use_raycast_visibility
        self.use_exact_volume = use_exact_volume
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
        self.rocks = []
        
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
        
        # Configure GPU usage in Preferences
        try:
            cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
            # Try OPTIX first, then CUDA
            try:
                cycles_prefs.compute_device_type = 'OPTIX'
            except:
                cycles_prefs.compute_device_type = 'CUDA'
                
            cycles_prefs.get_devices()
            
            # Enable all GPU devices
            print("Configuring Render Devices:")
            for device in cycles_prefs.devices:
                if device.type in {'CUDA', 'OPTIX'}:
                    device.use = True
                    print(f"  - Enabled: {device.name} ({device.type})")
                else:
                    device.use = False
                    
        except Exception as e:
            print(f"Warning: Failed to configure GPU preferences: {e}")
        
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

    def load_templates(self):
        # Load templates from library
        filepath = os.path.join(self.output_dir, "..", "rock_library.blend")
        if not os.path.exists(filepath):
            print(f"Error: Library not found at {filepath}")
            return []
            
        print(f"Loading templates from {filepath}...")
        
        # Append objects
        # accessing "Object" directory in blend file
        with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
            # Load all objects ending with "TemplateRock_"
            # actually we named them TemplateRock_X
            data_to.objects = [name for name in data_from.objects if name.startswith("TemplateRock")]
            
        # Link to collection
        templates = []
        for obj in data_to.objects:
            if obj is not None:
                # We don't link them to the scene to keep them hidden or just keep them as data?
                # If we want to instance them, we need them in data.
                # If we link them to scene they are visible. 
                # Let's instance them.
                # Just keep them in data_to.objects
                templates.append(obj)
                
        print(f"Loaded {len(templates)} templates.")
        return templates

    def spawn_rocks_direct(self, templates, num_rocks):
        rocks = []
        
        # Define spawn area (flat on belt)
        x_range = (-4.0, 4.0)
        y_range = (-4.0, 4.0)
        z_base = 0.05  # Just above belt
        
        # Size distribution: mix of small, medium, and large
        # Using a distribution that favors smaller but includes larger rocks
        def get_random_scale():
            r = random.random()
            if r < 0.8:  # 60% small
                return random.uniform(0.05, 0.25)
            elif r < 0.99:  # 30% medium
                return random.uniform(0.25, 0.35)
            else:  # 10% large
                return random.uniform(0.5, 1)
        
        for i in range(num_rocks):
            # Pick random template
            template = random.choice(templates)
            
            # Duplicate the template mesh data (not linked)
            new_mesh = template.data.copy()
            rock = bpy.data.objects.new(f"Rock_{i}", new_mesh)
            
            # Random position with slight jitter in Z for layering
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)
            z = z_base + random.uniform(0, 0.3)
            rock.location = (x, y, z)
            
            # Random rotation
            rock.rotation_euler = (
                random.uniform(0, math.pi * 2),
                random.uniform(0, math.pi * 2),
                random.uniform(0, math.pi * 2)
            )
            
            # Random scale with good distribution
            scale = get_random_scale()
            rock.scale = (scale, scale, scale)
            
            # Set pass index for segmentation
            rock.pass_index = i + 1
            
            # Link to scene
            self.scene.collection.objects.link(rock)
            rocks.append(rock)
        
        print(f"Generated {len(rocks)} rocks.")
        return rocks

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

        # Light
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        sun = bpy.context.active_object
        sun.data.energy = 5.0
        
        # Camera
        bpy.ops.object.camera_add(location=(0, 0, 7), rotation=(0, 0, 0))
        cam = bpy.context.active_object
        self.scene.camera = cam

    def get_camera_coords(self, cam, pos):
        """
        Project 3D world position to 2D camera coordinates (0-1).
        Returns (x, y) where (0,0) is bottom-left and (1,1) is top-right.
        Returns None if behind camera.
        """
        scene = self.scene
        # use dependency graph to ensure up to date
        coords = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
        
        # If z is negative, it's behind the camera
        if coords.z < 0:
            return None
            
        return Vector((coords.x, coords.y))

    def check_visibility(self, cam, obj, depsgraph):
        """
        Check if object is visible from camera using raycast.
        Cast rays to bounding box corners and center.
        """
        if not self.use_raycast_visibility:
            return True
        scene = self.scene
        cam_loc = cam.matrix_world.translation
        
        # Points to check: Center + Corners
        # Efficient check: just check center first?
        # For rocks, checking center might fail if center is occluded but edges are visible.
        # But for "size distribution", if most of it is occluded, maybe we don't want it?
        # Let's check center and random surface points.
        
        # Get evaluated mesh for accurate raycast
        obj_eval = obj.evaluated_get(depsgraph)
        
        # Simple approach: Check bounding box corners + center
        points = [Vector(p) for p in obj.bound_box] # 8 corners
        points.append(Vector((0,0,0))) # Center (local)
        
        # Transform to world
        world_points = [obj.matrix_world @ p for p in points]
        
        visible_points = 0
        total_checks = 0
        
        for p in world_points:
            # Direction from camera to point
            direction = p - cam_loc
            distance = direction.length
            direction.normalize()
            
            # Raycast
            # ray_cast(origin, direction, distance=1.70141e+38, depsgraph=None)
            # Returns (result, location, normal, index, object, matrix)
            result, location, normal, index, hit_obj, matrix = scene.ray_cast(depsgraph, cam_loc, direction, distance=distance - 0.01) # Bias to avoid self-hit at target?
            
            # If hit_obj is None, nothing blocked it (up to distance?).
            # Actually, ray_cast checks for *any* hit.
            # If we hit something *before* the target point, it is occluded.
            # We set distance limit to actual distance. 
            # If result is True, we hit something closer.
            # We need to check if hit_obj is the target object (or part of it).
            
            # If !result, we didn't hit anything in between? 
            # Wait, if we cast to the exact point on the object, we SHOULD hit the object.
            # But we might hit another object first.
            
            if not result:
                # No hit? Means clear line of sight? 
                # Or means we didn't hit anything (e.g. if point is inside geometry?)
                # If target point is on surface, we should hit it or something occluding it.
                # If target is center (inside), we must hit surface of own object first.
                
                # Let's rely on `hit_obj`. 
                # We cast slightly past the object? No.
                # Standard visibility check: cast to point.
                # If hit_obj is `obj`, or None (implying reached target?), it's visible.
                # Actually, `ray_cast` returns the *first* hit.
                pass
            
            if result:
                 # Check if hit object is us
                 if hit_obj.name == obj.name:
                     visible_points += 1
                 else:
                     # Hit something else (occluded)
                     pass
                     
        # If at least one point is visible, we count it.
        # Or maybe require a threshold?
        if visible_points > 0:
            return True
            
        return False

    def save_annotations(self, filename_prefix, frame_idx):
        """
        Generate and save YOLO labels (.txt) and detailed metadata (.json).
        """
        cam = self.scene.camera
        
        # Ensure depsgraph is up to date for correct positions
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        labels_yolo = []
        metadata = []
        
        # Get all rocks
        # dependent on how we track them. We can iterate scene objects and check name
        rocks = self.rocks if self.rocks else [obj for obj in self.scene.objects if obj.name.startswith("Rock")]
        
        image_width = self.scene.render.resolution_x
        image_height = self.scene.render.resolution_y
        
        for rock in rocks:
            # Check visibility first
            if not self.check_visibility(cam, rock, depsgraph):
                continue

            # We need the evaluated object to get the actual mesh data (optional for bbox, but good for volume)
            rock_eval = rock.evaluated_get(depsgraph)
            mesh = rock_eval.data
            
            # --- 2D Bounding Box ---
            # Project all 8 corners of the bounding box
            # rock.bound_box is in object space. Transform to world, then to camera.
            min_x, max_x = 1.0, 0.0
            min_y, max_y = 1.0, 0.0
            
            valid_points = False
            
            for corner in rock.bound_box:
                world_corner = rock.matrix_world @ Vector(corner)
                cam_coord = self.get_camera_coords(cam, world_corner)
                
                if cam_coord:
                    valid_points = True
                    min_x = min(min_x, cam_coord.x)
                    max_x = max(max_x, cam_coord.x)
                    min_y = min(min_y, cam_coord.y)
                    max_y = max(max_y, cam_coord.y)
            
            if not valid_points:
                continue
                
            # Clamp to image bounds
            min_x = max(0.0, min(1.0, min_x))
            max_x = max(0.0, min(1.0, max_x))
            min_y = max(0.0, min(1.0, min_y))
            max_y = max(0.0, min(1.0, max_y))
            
            # If off screen or too small
            if (max_x - min_x) < 0.001 or (max_y - min_y) < 0.001:
                continue
                
            # YOLO Format: class_id center_x center_y width height (normalized)
            # Origin is top-left for YOLO? No, YOLO is center_x, center_y relative to image.
            # Usually (0,0) is top-left in image coords, but Blender (0,0) is bottom-left?
            # Blender world_to_camera_view: (0,0) is bottom-left.
            # YOLO expects y from top? 
            #   Usually standard YOLO: x_center, y_center, w, h. 
            #   If image is processed by OpenCV, (0,0) is top-left.
            #   If blender gives y=0 at bottom, then y_opencv = 1 - y_blender.
            
            # Let's assume standard image coords (0,0 top-left).
            # Blender y: 0 (bottom) -> 1 (top).
            # Image y: 0 (top) -> 1 (bottom).
            # So y_img = 1.0 - y_blend.
            
            # Convert min/max Y to image coords (flip)
            # min_y_blend corresponds to max_y_img (bottom of object is high y in image)
            # max_y_blend corresponds to min_y_img (top of object is low y in image)
            
            top = 1.0 - max_y 
            bottom = 1.0 - min_y
            left = min_x
            right = max_x
            
            # center
            cx = (left + right) / 2.0
            cy = (top + bottom) / 2.0
            w = right - left
            h = bottom - top
            
            # Class ID 0 for Rock
            labels_yolo.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
            # --- Physical Metrics ---
            dims = rock.dimensions # x, y, z size in world units
            if self.use_exact_volume:
                import bmesh
                bm = bmesh.new()
                bm.from_mesh(mesh)
                vol_local = bm.calc_volume()
                bm.free()
                scale_fac = rock.scale.x * rock.scale.y * rock.scale.z
                vol_world = vol_local * scale_fac
            else:
                vol_world = dims.x * dims.y * dims.z
            
            metadata.append({
                "id": rock.pass_index, # Matches mask color/index
                "visible": True,
                "bbox_2d": [left, top, right, bottom], # Normalized [x1, y1, x2, y2]
                "bbox_yolo": [float(cx), float(cy), float(w), float(h)],
                "volume": float(vol_world),
                "dimensions": [float(dims.x), float(dims.y), float(dims.z)],
                "location": [float(rock.location.x), float(rock.location.y), float(rock.location.z)]
            })
            
        # Save YOLO Labels
        # Ensure labels dir
        labels_dir = os.path.join(self.output_dir, "labels")
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
            
        txt_path = os.path.join(labels_dir, f"{filename_prefix}.txt")
        with open(txt_path, 'w') as f:
            f.write("\n".join(labels_yolo))
            
        # Save JSON Metadata
        json_path = os.path.join(labels_dir, f"{filename_prefix}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "filename": f"{filename_prefix}.png",
                "width": image_width,
                "height": image_height,
                "objects": metadata
            }, f, indent=2)
    def run(self):
        print("Starting Dataset Generation...")
        
        # Load templates ONCE at the start
        templates = self.load_templates()
        if not templates:
            print("Error: Could not load templates")
            return
            
        for i in range(self.num_images):
            print(f"Generating Image {i+1}/{self.num_images}")
            
            self.clear_scene()
            self.setup_scene()
            
            # Spawn Rocks - reduced count for speed, direct placement for better distribution
            num_rocks = random.randint(5000, 15000) 
            print(f"Spawning {num_rocks} rocks...")
            
            self.rocks = self.spawn_rocks_direct(templates, num_rocks)

            # Set frame for rendering (no physics simulation)
            self.scene.frame_set(1)
            
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
            
            # Define Multiple Views
            # View 0: Top (Standard)
            # View 1: Angled Front
            # View 2: Angled Side
            # View 3: Iso
            
            views = [
                {"name": "top",   "pos": (0, 0, 9), "rot": (0, 0, 0)},
                {"name": "front", "pos": (0, -6, 7), "rot": (0.6, 0, 0)}, # Rotated roughly 35 deg X
                {"name": "side",  "pos": (6, 0, 7), "rot": (0.6, 0, 1.57)}, # Rotated X and Z
                {"name": "iso",   "pos": (5, 5, 8), "rot": (0.5, 0, 2.35)}  
            ]
            
            for v_idx, view in enumerate(views):
                # Set Camera
                self.scene.camera.location = view["pos"]
                self.scene.camera.rotation_euler = view["rot"]
                
                # Update output filename
                # Output node doesn't allow easy dynamic suffix changing per view within one frame.
                # But we can change the slot path.
                
                # Prefix format: rgb_{sim_id}_{view}_{frame} 
                # e.g. rgb_0001_top_0100
                
                img_prefix = f"rgb_{i:04d}_{view['name']}_"
                mask_prefix = f"mask_{i:04d}_{view['name']}_"
                
                out_node = self.scene.node_tree.nodes.get("File Output")
                out_node.file_slots[0].path = img_prefix
                out_node.file_slots[1].path = mask_prefix
                
                # Render
                print(f"Rendering View: {view['name']}")
                bpy.ops.render.render(write_still=False) 
                
                # Save Annotations
                final_frame_str = "0001"
                label_filename = f"{img_prefix}{final_frame_str}" 
                
                print(f"Saving annotations for {label_filename}...")
                self.save_annotations(label_filename, i)
            
        print("Dataset Generation Complete.")

if __name__ == "__main__":
    # Example usage: blender --background --python generate_dataset_blender.py
    gen = MiningConveyorSimulator(output_dir="/home/boom/startup/ParticleSim/dataset", num_images=5)
    gen.run()
