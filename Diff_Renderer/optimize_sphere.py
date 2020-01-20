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


folder_name= 'Uranus_lr4_Color'

#%%
material_map, mesh_list, light_map = pyredner.load_obj('sphere.obj')
# The teapot we loaded is relatively low-poly and doesn't have vertex normal.
# Fortunately we can compute the vertex normal from the neighbor vertices.
# We can use pyredner.compute_vertex_normal for this task:
# (Try commenting out the following two lines to see the differences in target images!)
for _, mesh in mesh_list:
    mesh.normals = pyredner.compute_vertex_normal(mesh.vertices/10, mesh.indices)
    print (_) # None
    
    
diffuse_reflectance =torch.tensor([1.0, 1.0, 0.0], device = pyredner.get_device())

mat_grey = pyredner.Material(diffuse_reflectance)

# The material list of the scene # 
materials = [mat_grey]


# Now we build a list of shapes using the list loaded from the Wavefront object file.
# Meshes loaded from .obj files may have different indices for uvs and normals,
# we use mesh.uv_indices and mesh.normal_indices to access them.
# This mesh does not have normal_indices so the value is None.

    
shape_sphere = pyredner.Shape(vertices = mesh.vertices/10,
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
shapes = [shape_sphere, shape_light]


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
img = render(1, *scene_args)

#%
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/init_guess.exr')
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/init_guess.png')

#Loading it again
target = pyredner.imread('results/'+folder_name+'/init_guess.exr')
target_p = pyredner.imread('results/'+folder_name+'/init_guess.png')
if pyredner.get_use_gpu():
    target = target.cuda()    
    
#%%    
im = Image.open("results/uranus.jpg")
#imm= transforms.CenterCrop((256,256))(im)
#im= im.resize((256,256))


im= transforms.Resize((256,256), interpolation= Image.BILINEAR)(im)

target= to_tensor(im)
target_np= torch_to_np(target)

target_np= reverse_channels(target_np)

#target_np[np.where(target_np==1)]=0
#inv_target_np = target_np-1

#plt.imshow(target_np)
target_3_channels = target_np[:,:,:3]

target= np_to_torch(target_3_channels)

#target = pyredner.imread('results/demo_triangle.jpg')

#torchvision.transforms.Resize(size, interpolation=2)

if pyredner.get_use_gpu():
    target = target.cuda()

#Next we want to produce the initial guess. We do this by perturb the scene.
#  Here I will check if we can load any image of a triangle and optimize our guess to the shape of the
#target img
     
# Here we change the vertices of the triangle from changing shape_traingle.vertices  
#This is how you change the vertices of a shape (ie a mesh in this case)

#Additional Line added to optimize this vertices
    
shape_sphere.vertices.requires_grad=True
diffuse_reflectance.requires_grad= True    
#shape_triangle.vertices = torch.tensor(\
#    [[-2.0,1.5,0.3], [0.9,1.2,-0.3], [-0.4,-1.4,0.2]],
#    device = pyredner.get_device(),
#    requires_grad = True)  # Becasue these values have to be optimized ie why  requires_grad = True

#%% from here on to the loop starts, this is only to print/show the init guess image
# It has nothing to do with the actual rendering of the init img or the optimization
# ie we run the scene_args again inisde the loop and render the image

#We need to serialize the scene again to get the new arguments. We then render our initial guess
#scene_args = pyredner.RenderFunction.serialize_scene(\
#    scene = scene,
#    num_samples = 16,
#    max_bounces = 1)

#img = render(1, *scene_args)   #img is the init guess image # img.shape is torch.Size([256, 256, 3])

# render takes a number as first argument 
#Check what this number is? Becasue in the final case we give 202 as the final argument to the render

# Save the image
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/init.png')
# Compute the difference and save the images.
diff = torch.abs(target - img)   # difference between the rgb values
pyredner.imwrite(diff.cpu(), 'results/'+folder_name+'/init_diff.png')


#%%
#Now we want to refine the initial guess using gradient-based optimization. 
#We use PyTorch's optimizer to do this.

#init lr was lr=5e-2
optimizer = torch.optim.Adam([shape_sphere.vertices, diffuse_reflectance], lr=5e-4) # what is to be optimized

#Here we need to add a line or as above that we want to optimize colors as well, ie the material /diffuse_reflectance

for t in range(2700):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4, # We use less samples in the Adam loop.
        max_bounces = 1)
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    img = render(t+1, *scene_args) # rednering of the initial guess
    # Save the intermediate render.
    if(t%5==0 or t==2699):
        pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()   # simple l2 loss between the rgb values of the image
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients of the three vertices.
    print('grad for vertices:', shape_sphere.vertices.grad)
    print('grad for color:', diffuse_reflectance.grad)
    # Take a gradient descent step.
    optimizer.step()
    # Print the current three vertices.
    print('vertices:', shape_sphere.vertices)
    print('diffuse_reflectance ie Color', diffuse_reflectance)

#%
# Render the final result.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
img = render(2702, *scene_args)
# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/final.exr')
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/'+folder_name+'/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/'+folder_name+'/iter_%d.png", "-vb", "20M",
    "results/'+folder_name+'/out.mp4"])
    
    
    
    
    
    
    
    
    
    
    