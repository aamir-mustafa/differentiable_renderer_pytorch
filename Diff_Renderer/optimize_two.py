#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:05:02 2019

@author: am2806
"""
#import plotly.graph_objects as go
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
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

pyredner.set_use_gpu(torch.cuda.is_available())
#pyredner.set_use_gpu(False)

#First, we setup a camera, by constructing a pyredner.Camera object 

# I think this is known and not optimized

cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (1000, 2000),
                      fisheye = False)


folder_name= 'Cylinder_Sphere_Cuda'

#%
material_map, mesh_list, light_map = pyredner.load_obj('ReferenceOutputMeshes/cylinderHighNVO.obj')

material_map1, mesh_list1, light_map1 = pyredner.load_obj('sphere.obj')

#%%
# The teapot we loaded is relatively low-poly and doesn't have vertex normal.
# Fortunately we can compute the vertex normal from the neighbor vertices.
# We can use pyredner.compute_vertex_normal for this task:
# (Try commenting out the following two lines to see the differences in target images!)
for _, mesh in mesh_list:
    mesh.normals = pyredner.compute_vertex_normal(mesh.vertices/1.5, mesh.indices)
    print (_) # None
    
for _, mesh1 in mesh_list1:
    mesh1.normals = pyredner.compute_vertex_normal(mesh1.vertices/25, mesh1.indices)
    print (_)

    
diffuse_reflectance =torch.tensor([0.0, 1.0, 0.4], device = pyredner.get_device())

mat_grey = pyredner.Material(diffuse_reflectance)

# The material list of the scene # 
materials = [mat_grey]


shape_cylinder = pyredner.Shape(vertices = mesh.vertices/1.5,
                               indices = mesh.indices,
                               uvs = None, normals = mesh.normals, material_id = 0)

shape_sphere= pyredner.Shape(vertices = mesh1.vertices/25,
                               indices = mesh1.indices,
                               uvs = None, normals = mesh1.normals, material_id = 0)

    
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
shapes = [shape_sphere, shape_cylinder ]
#shapes = [shape_sphere, shape_cylinder , shape_light]

#
#light = pyredner.AreaLight(shape_id = 1, 
#                           intensity = torch.tensor([200.0,200.0,200.0]))
#area_lights = [light]
## Finally we construct our scene using all the variables we setup previously.
#scene = pyredner.Scene(cam, shapes, materials, area_lights)

#%%


envmap = pyredner.imread('sunsky.exr')
if pyredner.get_use_gpu():
    envmap = envmap.cuda()
envmap = pyredner.EnvironmentMap(envmap)

scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = envmap)

# Like the previous tutorial, we serialize and render the scene, 
# Like the previous tutorial, we serialize and render the scene, 
# save it as our target
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)

render = pyredner.RenderFunction.apply
img = render(0, *scene_args)    # gives an exr image


#%
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/target.exr')
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/target.png')

#Loading it again
target_p = pyredner.imread('results/'+folder_name+'/target.png')
target = pyredner.imread('results/'+folder_name+'/target.exr')

if pyredner.get_use_gpu():
    target = target.cuda()    
    
#%%    
#im = Image.open("results/circle_1.png")
##imm= transforms.CenterCrop((256,256))(im)
##im= im.resize((256,256))
#
#
#im= transforms.Resize((256,256), interpolation= Image.BILINEAR)(im)
#
#target= to_tensor(im)
#target_np= torch_to_np(target)
#
#target_np= reverse_channels(target_np)
#plt.imshow(target_np)
#
##target_np[np.where(target_np==1)]=0
##inv_target_np = target_np-1
#
##plt.imshow(target_np)
#target_3_channels = target_np[:,:,:3]
#
#t= target_3_channels
#
#for i in range(256):
##    for j in range(256):
#    if t[i,i].all()== 1.0:
#        t[i,i]=[0.0, 0.0, 0.0]
#
#plt.imshow(t)
#target= np_to_torch(target_3_channels)
#
##target = pyredner.imread('results/demo_triangle.jpg')
#
##torchvision.transforms.Resize(size, interpolation=2)
#
#if pyredner.get_use_gpu():
#    target = target.cuda()


#%% 
#def gen_scale_matrix(scale):
##    o = torch.ones([1], dtype=torch.float32)
#    return torch.diag(torch.cat([scale, scale, scale], 0)) 

    
translation_params = torch.tensor([0.35, 0.0, 0.0], # [+ means left, + means up,means back ]
    device = pyredner.get_device(), requires_grad=True)
translation = translation_params * 3.0
euler_angles = torch.tensor([0.0, -0.0, 0.0], device = pyredner.get_device(), requires_grad=True)

scale = torch.tensor([0.9],
    device = pyredner.get_device() , requires_grad=True)
# We obtain the teapot vertices we want to apply the transformation on.
shape0_vertices = shapes[0].vertices.clone()

#shape1_vertices = shapes[1].vertices.clone()
# We can use pyredner.gen_rotate_matrix to generate 3x3 rotation matrices

rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
#scale_matrix= gen_scale_matrix(scale)

if pyredner.get_use_gpu():
    rotation_matrix = rotation_matrix.cuda()
#    scale_matrix = scale_matrix.cuda()
    
center = torch.mean(torch.cat([shape0_vertices]), 0)
# We shift the vertices to the center, apply rotation matrix,
# then shift back to the original space.
shapes[0].vertices = \
    ((shape0_vertices - center) *scale) @ torch.t(rotation_matrix) + \
    center + translation

translation_params1 = torch.tensor([-0.3, 0.0, 0.0], # [+ means left, + means up,means back ]
    device = pyredner.get_device(), requires_grad=True)
translation1 = translation_params1 * 3.0
euler_angles1 = torch.tensor([0.0, -0.0, 0.0],device = pyredner.get_device(),  requires_grad=True)

scale1 = torch.tensor([1.25],
    device = pyredner.get_device() , requires_grad=True)
# We obtain the teapot vertices we want to apply the transformation on.
shape1_vertices = shapes[1].vertices.clone()

#shape1_vertices = shapes[1].vertices.clone()
# We can use pyredner.gen_rotate_matrix to generate 3x3 rotation matrices

rotation_matrix1 = pyredner.gen_rotate_matrix(euler_angles1)
#scale_matrix= gen_scale_matrix(scale)

if pyredner.get_use_gpu():
    rotation_matrix1 = rotation_matrix1.cuda()
#    scale_matrix = scale_matrix.cuda()
    
center1 = torch.mean(torch.cat([shape1_vertices]), 0)
    
shapes[1].vertices = \
    (shape1_vertices - center1) @ torch.t(rotation_matrix1) + \
    center1 + translation1
# Since we changed the vertices, we need to regenerate the shading normals
shapes[0].normals = pyredner.compute_vertex_normal(shapes[0].vertices, shapes[0].indices)
shapes[1].normals = pyredner.compute_vertex_normal(shapes[1].vertices, shapes[1].indices)
# We need to serialize the scene again to get the new arguments.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
# Render the initial guess.
img = render(1, *scene_args)
# Save the images.
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/init.png')
# Compute the difference and save the images.
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/'+folder_name+'/init_diff.png')

#%%
# Optimize for pose parameters.
optimizer = torch.optim.Adam([translation_params, scale, euler_angles, translation_params1, scale1, euler_angles1], lr=1e-2)
# Run 200 Adam iterations.
import time
t1= time.time()


for t in range(300):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: apply the mesh operation and render the image.
    translation = translation_params * 3.0
    rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
    
    translation1 = translation_params1 * 3.0
    rotation_matrix1 = pyredner.gen_rotate_matrix(euler_angles1)    
#    scale_matrix= gen_scale_matrix(scale)
    
    if pyredner.get_use_gpu():
        rotation_matrix = rotation_matrix.cuda()
        rotation_matrix1 = rotation_matrix1.cuda()
        
    center = torch.mean(torch.cat([shape0_vertices]), 0)
    center1 = torch.mean(torch.cat([shape1_vertices]), 0)
    
    shapes[0].vertices = \
        ((shape0_vertices - center) *scale) @ torch.t(rotation_matrix) + \
        center + translation
    shapes[1].vertices = \
        (shape1_vertices - center1)*scale1 @ torch.t(rotation_matrix1) + \
        center1 + translation1
        
    shapes[0].normals = pyredner.compute_vertex_normal(shapes[0].vertices, shapes[0].indices)
    shapes[1].normals = pyredner.compute_vertex_normal(shapes[1].vertices, shapes[1].indices)
    
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4, # We use less samples in the Adam loop.
        max_bounces = 1)
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    img = render(t+302, *scene_args)
    # Save the intermediate render.
    if(t%25==0 or t==299):
        pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/iter_{}.png'.format(t+300))    
#    pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('translation_params.grad:', translation_params.grad)
    print('euler_angles.grad:', euler_angles.grad)
    print('scale.grad:', scale.grad)
    
    print('translation_params1.grad:', translation_params1.grad)
    print('euler_angles1.grad:', euler_angles1.grad)
    print('scale1.grad:', scale1.grad)
    
#    print('color.grad:', diffuse_reflectance.grad)
    # Take a gradient descent step.
    optimizer.step()
    # Print the current pose parameters.
    print('translation:', translation)
    print('euler_angles:', euler_angles)
    print('scale:', scale)
    
    print('translation1:', translation1)
    print('euler_angles1:', euler_angles1)
    print('scale1:', scale1)    
#    print('color/Diffuse Reflectance :', diffuse_reflectance)
    

# Render the final result.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
img = render(602, *scene_args)
# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/final.exr')
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/'+folder_name+'/final_diff.png')

# Convert the intermediate renderings to a video.
#from subprocess import call
#call(["ffmpeg", "-framerate", "24", "-i",
#    "results/"+folder_name+"/iter_%d.png", "-vb", "20M",
#    "results/"+folder_name+"/out.mp4"])

    
print ("Total time taken is {} seconds " .format(time.time() -t1))
#%%
translation = translation_params * 3.0
rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)

center = torch.mean(torch.cat([shape0_vertices]), 0)

shapes[0].vertices = \
        ((shape0_vertices - center) *scale) @ torch.t(rotation_matrix) + \
        center + translation

v_np= torch_to_np(shapes[0].vertices)

#Axes3D.plot_surface(X=v_np[:,0], Y=v_np[:,1], Z=v_np[:,2])

x,y,z =v_np[:,0],v_np[:,1], v_np[:,2]; 
   
dia=( (x.max()-x.min())+ (y.max()-y.min()) + (z.max()-z.min()))/3
print('Radius of the Sphere is:', dia/2)

xx,yy,zz= np.ravel(x),np.ravel(y),np.ravel(z)

fig=p.figure()
ax = p3.Axes3D(fig)
# plot3D requires a 1D array for x, y, and z
# ravel() converts the 100x100 array into a 1x10000 array
ax.plot3D(xx,yy,zz)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.add_axes(ax)
p.show()

xc,yc,zc= x.max()- dia/2, y.max()- dia/2, z.max()-dia/2
print('Centre of the Sphere is:', xc, yc, zc )
#%%    

#from numpy import *

v = mesh.vertices/5

v_np= torch_to_np(v)

#Axes3D.plot_surface(X=v_np[:,0], Y=v_np[:,1], Z=v_np[:,2])

x,y,z =v_np[:,0],v_np[:,1], v_np[:,2];

xx,yy,zz= np.ravel(x),np.ravel(y),np.ravel(z)

fig=p.figure()
ax = p3.Axes3D(fig)
# plot3D requires a 1D array for x, y, and z
# ravel() converts the 100x100 array into a 1x10000 array
ax.plot3D(np.ravel(x),np.ravel(y),np.ravel(z))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.add_axes(ax)
p.show()

#%% Scatter Plot for Mesh
#!python
fig=p.figure()
ax = p3.Axes3D(fig)
# scatter3D requires a 1D array for x, y, and z
# ravel() converts the 100x100 array into a 1x10000 array
ax.scatter3D(np.ravel(x),np.ravel(y),np.ravel(z))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
p.show()

#%%
translation= torch.tensor([ 0.1040, -7.4208, 31.6828])
euler_angles= torch.tensor([ 0.1000, -0.1000,  0.1000])
scale= torch.tensor([3.7331])


rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
    
center = torch.mean(torch.cat([v]), 0)

v_trans = \
    ((v - center) *scale) @ torch.t(rotation_matrix) + \
    center + translation

v_np= torch_to_np(v_trans)

#Axes3D.plot_surface(X=v_np[:,0], Y=v_np[:,1], Z=v_np[:,2])

x,y,z =v_np[:,0],v_np[:,1], v_np[:,2];

xx,yy,zz= np.ravel(x),np.ravel(y),np.ravel(z)

fig=p.figure()
ax = p3.Axes3D(fig)
# plot3D requires a 1D array for x, y, and z
# ravel() converts the 100x100 array into a 1x10000 array
ax.plot3D(x,y,z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.add_axes(ax)
p.show()

    
    
    
    