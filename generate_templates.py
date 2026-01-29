import bpy
import random
import os
import sys

# Add current dir to path if needed
sys.path.append(os.getcwd())

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Remove orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

def create_rock_material(name):
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

def create_template_rock(index):
    # Create a base cube
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0))
    obj = bpy.context.active_object
    obj.name = f"TemplateRock_{index}"
    
    # 1. Subdivision (Catmull-Clark)
    subsurf = obj.modifiers.new(name="Subsurf", type='SUBSURF')
    subsurf.subdivision_type = 'CATMULL_CLARK'
    subsurf.levels = 4
    
    # 2. Large Scale Displacement
    disp_shape = obj.modifiers.new(name="Displace_Shape", type='DISPLACE')
    tex_shape = bpy.data.textures.new(f"Texture_Shape_{index}", 'VORONOI')
    tex_shape.noise_scale = random.uniform(0.5, 1.5)
    tex_shape.distance_metric = 'DISTANCE' 
    disp_shape.texture = tex_shape
    disp_shape.strength = random.uniform(0.3, 0.7)
    
    # 3. Small Scale Displacement
    disp_detail = obj.modifiers.new(name="Displace_Detail", type='DISPLACE')
    tex_detail = bpy.data.textures.new(f"Texture_Detail_{index}", 'CLOUDS')
    tex_detail.noise_scale = random.uniform(2.0, 4.0) 
    disp_detail.texture = tex_detail
    disp_detail.strength = random.uniform(0.05, 0.15)
    
    # 4. Decimate
    decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
    decimate.ratio = random.uniform(0.05, 0.2)
    
    # Bake modifiers specifically using depsgraph
    # bpy.ops.object.modifier_apply is flaky in background
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_new = bpy.data.meshes.new_from_object(obj_eval)
    mesh_new.name = obj.data.name + "_baked"
    
    # Replace mesh
    old_mesh = obj.data
    obj.modifiers.clear()
    obj.data = mesh_new
    bpy.data.meshes.remove(old_mesh)
    
    # Set shading
    bpy.ops.object.shade_flat()
    
    # Assign Material
    mat = create_rock_material(f"Mat_{index}")
    obj.data.materials.append(mat)
    
    # Move to side
    obj.location.x = index * 2.0
    
    return obj

def main():
    print("Generating Rock Templates...")
    clean_scene()
    
    num_templates = 50
    for i in range(num_templates):
        create_template_rock(i)
        
    print(f"Created {num_templates} templates.")
    
    # Save the file
    save_path = os.path.join(os.getcwd(), "rock_library.blend")
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    print(f"Saved library to {save_path}")

if __name__ == "__main__":
    main()
