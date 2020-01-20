#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:05:02 2019

@author: am2806
"""

import pyredner
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def load(path):
    img = Image.open(path)
    return img

to_tensor = transforms.ToTensor()

to_pil = transforms.ToPILImage()

def torch_to_np(img_var):
    return img_var.detach().cpu().numpy()[:]

def np_to_torch(img_np):
    return torch.from_numpy(img_np)[ :]

def reverse_channels(img):
    return np.moveaxis(img, 0, -1) # source, dest

#pyredner.set_use_gpu(torch.cuda.is_available())
pyredner.set_use_gpu(False)

#First, we setup a camera, by constructing a pyredner.Camera object 

# I think this is known and not optimized

cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)




#%%
material_map, mesh_list, light_map = pyredner.load_obj('diamond.obj')
# The teapot we loaded is relatively low-poly and doesn't have vertex normal.
# Fortunately we can compute the vertex normal from the neighbor vertices.
# We can use pyredner.compute_vertex_normal for this task:
# (Try commenting out the following two lines to see the differences in target images!)
for _, mesh in mesh_list:
    mesh.normals = pyredner.compute_vertex_normal(mesh.vertices/20, mesh.indices)
    print (_) # None
    
    
diffuse_reflectance =torch.tensor([0.0, 1.0, 0.0], device = pyredner.get_device(), requires_grad=True)

mat_grey = pyredner.Material(diffuse_reflectance)

# The material list of the scene # 
materials = [mat_grey]


# Now we build a list of shapes using the list loaded from the Wavefront object file.
# Meshes loaded from .obj files may have different indices for uvs and normals,
# we use mesh.uv_indices and mesh.normal_indices to access them.
# This mesh does not have normal_indices so the value is None.

    
shape_diamond = pyredner.Shape(vertices = mesh.vertices/20,
                               indices = mesh.indices,
                               uvs = None, normals = mesh.normals, material_id = 0)
    
shape_light = pyredner.Shape(\
    vertices = torch.tensor([[-1.0, -1.0, -7.0],
                             [ 2.0, -1.0, -7.0],
                             [-1.0,  1.0, -7.0],
                             [ 2.0,  1.0, -7.0]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)
    
# The shape list of our scene contains two shapes:
shapes = [shape_diamond, shape_light]


light = pyredner.AreaLight(shape_id = 1, 
                           intensity = torch.tensor([200.0,200.0,200.0]))
area_lights = [light]
# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam, shapes, materials, area_lights)
    
#shapes = []
#for _, mesh in mesh_list:
#    assert(mesh.normal_indices is None)
#    shapes.append(pyredner.Shape(\
#        vertices = mesh.vertices,
#        indices = mesh.indices,
#        uvs = None,
#        normals = mesh.normals,
#        material_id=0))


# The previous tutorial used a mesh area light for the scene lighting, 
# here we use an environment light,
# which is a texture representing infinitely far away light sources in 
# spherical coordinates.
#envmap = pyredner.imread('sunsky.exr')
#if pyredner.get_use_gpu():
#    envmap = envmap.cuda()
#envmap = pyredner.EnvironmentMap(envmap)
#
## Finally we construct our scene using all the variables we setup previously.
#scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = envmap)
    
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)

render = pyredner.RenderFunction.apply
img = render(0, *scene_args)

#%%
pyredner.imwrite(img.cpu(), 'results/pose_estimation/target.exr')
pyredner.imwrite(img.cpu(), 'results/pose_estimation/target.png')

#Loading it again
target = pyredner.imread('results/pose_estimation/target.exr')

if pyredner.get_use_gpu():
    target = target.cuda()    
    
#%%    
    
    
    
    
    
    
    
    
    
    
    
    