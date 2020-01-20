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
print(torch.version.cuda)
from PIL import Image
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
#import kornia
#import torch.nn.functional as F
#from edge_detection_network import Network

def load(path):
    img = Image.open(path)
    return img

to_tensor = transforms.ToTensor()  #Convert a PIL Image or numpy.ndarray to tensor.

to_pil = transforms.ToPILImage()   #Convert a tensor or an ndarray to PIL Image

def torch_to_np(img_var):
    return img_var.detach().cpu().numpy()[:]

def np_to_torch(img_np):
    return torch.from_numpy(img_np)[ :]

def reverse_channels(img):
    return np.moveaxis(img, 0, -1) # source, dest


def channels_first_torch(img_torch):
    img_torch= torch.transpose(img_torch, 0, -1)
    return torch.transpose(img_torch, 1, -1)

def channels_last_torch(img_torch):
    return torch.transpose(img_torch, 0, -1)

pyredner.set_use_gpu(torch.cuda.is_available())
#pyredner.set_use_gpu(False)
print ('Cuda Available:', torch.cuda.is_available())
#pyredner.set_use_gpu(False)

#First, we setup a camera, by constructing a pyredner.Camera object 


#moduleNetwork = Network().cuda().eval()
#
#def estimate(tensorInput):
#	intWidth = tensorInput.size(2)
#	intHeight = tensorInput.size(1)
#
##	assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
##	assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
#
#	return moduleNetwork(tensorInput.cuda().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()


cam = pyredner.Camera(position = torch.tensor([0.0, -0.0, -5.0]),  # -8.5
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (1650, 2843),
                      fisheye = False)


folder_name= 'HDR_Cube_2'

#%
#material_map, mesh_list, light_map = pyredner.load_obj('ReferenceOutputMeshes/cylinderHighNVO.obj')
#material_map1, mesh_list1, light_map1 = pyredner.load_obj('hemisphere.obj')

material_map2, mesh_list2, light_map2 = pyredner.load_obj('ReferenceOutputMeshes/cubeNVO.obj')

#material_map3, mesh_list3, light_map3 = pyredner.load_obj('cone.obj')
#%%
# The teapot we loaded is relatively low-poly and doesn't have vertex normal.
# Fortunately we can compute the vertex normal from the neighbor vertices.
# We can use pyredner.compute_vertex_normal for this task:
# (Try commenting out the following two lines to see the differences in target images!)

#for _, mesh in mesh_list:
#    mesh.normals = pyredner.compute_vertex_normal(mesh.vertices/2.5, mesh.indices)
#    print (_) # None
    
#for _, mesh1 in mesh_list1:
#    mesh1.normals = pyredner.compute_vertex_normal(mesh1.vertices/1, mesh1.indices)
#    print (_)

for _, mesh2 in mesh_list2:
    mesh2.normals = pyredner.compute_vertex_normal(mesh2.vertices/3, mesh2.indices)

#for _, mesh3 in mesh_list3:
#    mesh3.normals = pyredner.compute_vertex_normal(mesh3.vertices/5, mesh3.indices)
    
    
#diffuse_reflectance_green =torch.tensor([0.0, 1.0, 0.0], device = pyredner.get_device())
diffuse_reflectance_green =torch.tensor([0.65, 0.32, 0.16], device = pyredner.get_device())
mat_green = pyredner.Material(diffuse_reflectance_green)

#diffuse_reflectance_red =torch.tensor([1.0, 0.0, 0.0], device = pyredner.get_device())
diffuse_reflectance_red =torch.tensor([0.65, 0.32, 0.16], device = pyredner.get_device())
mat_red = pyredner.Material(diffuse_reflectance_red)

#diffuse_reflectance_blue =torch.tensor([0.0, 0.0, 1.0], device = pyredner.get_device())
diffuse_reflectance_blue =torch.tensor([0.65, 0.32, 0.16], device = pyredner.get_device())
mat_blue = pyredner.Material(diffuse_reflectance_blue)

#diffuse_reflectance_purple =torch.tensor([0.50, 0.0, 0.50], device = pyredner.get_device())
diffuse_reflectance_purple =torch.tensor([0.65, 0.32, 0.16], device = pyredner.get_device())
mat_purple = pyredner.Material(diffuse_reflectance_purple)


# The material list of the scene # 
#materials = [mat_green, mat_red, mat_blue, mat_purple]

materials = [mat_green]

#shape_cylinder = pyredner.Shape(vertices = mesh.vertices/2.5,
#                               indices = mesh.indices,
#                               uvs = None, normals = mesh.normals, material_id = 0)

#shape_hemisphere= pyredner.Shape(vertices = mesh1.vertices/1,
#                               indices = mesh1.indices,
#                               uvs = None, normals = mesh1.normals, material_id = 1)

shape_cube = pyredner.Shape(vertices = mesh2.vertices/3,
                               indices = mesh2.indices,
                               uvs = None, normals = mesh2.normals, material_id = 0) # was 2

#shape_cone = pyredner.Shape(vertices = mesh3.vertices/5,
#                               indices = mesh3.indices,
#                               uvs = None, normals = mesh3.normals, material_id = 3)
    
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

#shape_light_top = pyredner.Shape(\
#    vertices = torch.tensor([[-1.9, 12.0, -2.25],
#                             [ 1.9, 12.0, -2.25],
#                             [-1.9,  12.0, 2.25],
#                             [ 1.9,  12.0, 2.25]], device = pyredner.get_device()),
#    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
#        dtype = torch.int32, device = pyredner.get_device()),
#    uvs = None,
#    normals = None,
#    material_id = 0)
    
# The shape list of our scene contains two shapes:
#shapes = [shape_cylinder, shape_hemisphere, shape_cube, shape_cone, shape_light_top ]
    
#shapes = [shape_cube, shape_light]
shapes = [ shape_light]

#light = pyredner.AreaLight(shape_id = 4, 
#                           intensity = torch.tensor([200.0,200.0,200.0]))


light = pyredner.AreaLight(shape_id = 0, 
                           intensity = torch.tensor([200.0,200.0,200.0]))
area_lights = [light]

## Finally we construct our scene using all the variables we setup previously.
#scene = pyredner.Scene(cam, shapes, materials, area_lights)

#%

envmap = pyredner.imread('sunsky.exr')
#envmap = pyredner.imread('results/hdr_image_cropped.exr')
#envmap = 1/10*(torch.log2(envmap*(2**24)+1))

#envmap= envmap*5     #the value chosen visually
#envmap= transforms.Resize((64,64), interpolation=2)(to_pil(envmap))
#envmap= to_tensor(envmap)
#envmap=channels_last_torch(envmap)
if pyredner.get_use_gpu():
    envmap = envmap.cuda()
envmap = pyredner.EnvironmentMap(envmap)

scene = pyredner.Scene(cam, shapes, materials, area_lights = [light], envmap = envmap)

# Like the previous tutorial, we serialize and render the scene, 
# Like the previous tutorial, we serialize and render the scene, 
# save it as our target
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16, #512
    max_bounces = 1)
#
render = pyredner.RenderFunction.apply
img = render(0, *scene_args)    # gives an exr image


#%
#pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/target.exr')
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/init_load_top.png') # saves an exr image as png
#%%
##Loading it again
#target_p = pyredner.imread('results/'+folder_name+'/target.png')

#target = pyredner.imread('results/'+folder_name+'/target.exr')
target = pyredner.imread('Target_Images_Cropped/cube_front_cropped.exr')
#
if pyredner.get_use_gpu():
    target = target.cuda()    
pyredner.imwrite(target.cpu(), 'results/'+folder_name+'/target.png')    #saves an exr image as png
#%
  
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


#%
#def gen_scale_matrix(scale):
##    o = torch.ones([1], dtype=torch.float32)
#    return torch.diag(torch.cat([scale, scale, scale], 0)) 

# Cylinder    

#translation_params = torch.tensor([-1.0, -1.0, 0.0], # [+ means left, + means up, + means back ]
translation_params = torch.tensor([-0.0, -0.0, 0.0],                                  
    device = pyredner.get_device(), requires_grad=True)
translation = translation_params * 1.0
euler_angles = torch.tensor([0.0, -0.0, 0.0], device = pyredner.get_device(), requires_grad=False)

scale = torch.tensor([1.0],
    device = pyredner.get_device() , requires_grad=True)
# We obtain the teapot vertices we want to apply the transformation on.
shape0_vertices = shapes[0].vertices.clone()

rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)

if pyredner.get_use_gpu():
    rotation_matrix = rotation_matrix.cuda()
    
center = torch.mean(torch.cat([shape0_vertices]), 0)
shapes[0].vertices = \
    ((shape0_vertices - center) *scale) @ torch.t(rotation_matrix) + \
    center + translation

# Hemisphere

#translation_params1 = torch.tensor([1.1, -0.5, 0.0], # [+ means left, + means up,+ means back ]
#    device = pyredner.get_device(), requires_grad=True)
#translation1 = translation_params1 * 1.0
#euler_angles1 = torch.tensor([0.0, 0.750, -1.075],device = pyredner.get_device(),  requires_grad=False) #[,+ means forwards turn, + means anti clockwise]
#
#scale1 = torch.tensor([0.60],
#    device = pyredner.get_device() , requires_grad=True)
#
#shape1_vertices = shapes[1].vertices.clone()
#
#
#rotation_matrix1 = pyredner.gen_rotate_matrix(euler_angles1)
#
#if pyredner.get_use_gpu():
#    rotation_matrix1 = rotation_matrix1.cuda()
#    
#center1 = torch.mean(torch.cat([shape1_vertices]), 0)
#shapes[1].vertices = \
#    (shape1_vertices - center1) *scale1 @ torch.t(rotation_matrix1) + \
#    center1 + translation1

#Cube

#translation_params2 = torch.tensor([0.8, -0.9, 0.0], # [+ means left, + means up, + means back ]
#    device = pyredner.get_device(), requires_grad=True)
#translation2 = translation_params2 * 1.0
#euler_angles2 = torch.tensor([0.0, -0.0, 0.0],device = pyredner.get_device(),  requires_grad=False)
#
#scale2 = torch.tensor([1.0],
#    device = pyredner.get_device() , requires_grad=True)
#
#shape2_vertices = shapes[2].vertices.clone()
#rotation_matrix2 = pyredner.gen_rotate_matrix(euler_angles2)
#
#if pyredner.get_use_gpu():
#    rotation_matrix2 = rotation_matrix2.cuda()
#    
#center2 = torch.mean(torch.cat([shape2_vertices]), 0)
#shapes[2].vertices = \
#    (shape2_vertices - center2)*scale2 @ torch.t(rotation_matrix2) + \
#    center2 + translation2    
    
#Cone
    
#translation_params3 = torch.tensor([-0.7, -1.40, -0.0], # [+ means left, + means up, + means back ]
#    device = pyredner.get_device(), requires_grad=True)
#translation3 = translation_params3 * 1.0
#euler_angles3 = torch.tensor([0.0, -0.0, 0.0], device = pyredner.get_device(),  requires_grad=False)
#
#scale3 = torch.tensor([1.2],
#    device = pyredner.get_device() , requires_grad=True)
#
#shape3_vertices = shapes[3].vertices.clone()
#rotation_matrix3 = pyredner.gen_rotate_matrix(euler_angles3)
#
#if pyredner.get_use_gpu():
#    rotation_matrix3 = rotation_matrix3.cuda()
#    
#center3 = torch.mean(torch.cat([shape3_vertices]), 0)
#shapes[3].vertices = \
#    (shape3_vertices - center3)*scale3 @ torch.t(rotation_matrix3) + \
#    center3 + translation3  
    
    
shapes[0].normals = pyredner.compute_vertex_normal(shapes[0].vertices, shapes[0].indices)
#shapes[1].normals = pyredner.compute_vertex_normal(shapes[1].vertices, shapes[1].indices)
#shapes[2].normals = pyredner.compute_vertex_normal(shapes[2].vertices, shapes[2].indices)
#shapes[3].normals = pyredner.compute_vertex_normal(shapes[3].vertices, shapes[3].indices)

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16, #512
    max_bounces = 1)
# Render the initial guess.
img = render(1, *scene_args)
# Save the images.
normalization= torch.tensor([img[:,:,0].max(), img[:,:,1].max(), img[:,:,2].max()], device = pyredner.get_device())

pyredner.imwrite((img/normalization).cpu(), 'results/'+folder_name+'/init_top_transformed.png')
# Compute the difference and save the images.
normalization= torch.tensor([img[:,:,0].max(), img[:,:,1].max(), img[:,:,2].max()], device = pyredner.get_device())
diff = torch.abs(target - img/normalization)
pyredner.imwrite(diff.cpu(), 'results/'+folder_name+'/init_diff.png')

#%% Computing Difference Edges

#edge_img= channels_first_torch(img/normalization)
#img_init=  255*torch.log10(img/normalization +1)
#img_init = img_init.cpu().detach().numpy()
#plt.imshow(img_init)
#
#plt.imshow(target.cpu().detach().numpy())
#
#target_log = torch.log2(target*(2**24)+1)
#target_log = target_log.cpu().detach().numpy()
#normalization= np.array([target_log[:,:,0].max(), target_log[:,:,1].max(), target_log[:,:,2].max()])
##
#plt.imshow(target_log/normalization)


#target_log = 255*torch.log10(target +1)
#target_log = target_log.cpu().detach().numpy()
#plt.imshow(target_log)


#plt.imsave('results/'+folder_name+'/demo.png', target_log.cpu().detach().numpy())
#
#
#edge_img= channels_first_torch(target)
#x_gray = kornia.rgb_to_grayscale(edge_img.float())
#
#edge_img= x_gray[None,:,:,:]
#edges_guess= kornia.sobel(edge_img)
#
#
#edges_guess_np1=(edges_guess[0][0]).cpu().detach().numpy()
#plt.imshow(edges_guess_np1, cmap='gray')
#plt.imsave('results/'+folder_name+'/demo.png',edges_guess_np1, cmap='gray')
#
#a=edges_guess_np1
#a[a>=0.0001]=1.0
#plt.imshow(a, cmap='gray')
#plt.imsave('results/'+folder_name+'/demo.png',a, cmap='gray')
#for i in range(2291):
#    for j in range(2291):
#        if (a[i,j]!=0):
#            a[i,j]=1.0

#filter = torch.tensor([[1., 2. , 1.], [0., 0., 0.], [-1., -2. , -1.]],device = pyredner.get_device())
#f = filter.expand(1,3,3,3)
#
#sobel= F.conv2d(edge_img,f, stride=1, padding=1)
#
#edges_guess_np1=(sobel[0][0]).cpu().detach().numpy()
#edges_guess_np1= edges_guess_np1- edges_guess_np1.min()
#edges_guess_np1= edges_guess_np1/(edges_guess_np1.max()- edges_guess_np1.min())
#plt.imshow(edges_guess_np1)

#edge_target= channels_first_torch(target)
#edge_target= edge_target[None,:,:,:]
#edges_target_img= kornia.sobel(edge_target)
#
##For plotting
#edges_guess_np=channels_last_torch(edges_guess[0]).cpu().detach().numpy()
#plt.imshow(edges_guess_np)
#normalization= np.array([edges_guess_np[:,:,0].max(), edges_guess_np[:,:,1].max(), edges_guess_np[:,:,2].max()])
#
#plt.imsave('results/'+folder_name+'/demo.png', edges_guess_np/normalization)
#
#plt.imshow(channels_last_torch(edges_target_img[0]).cpu().detach().numpy())
#%
# Optimize for pose parameters.
#optimizer = torch.optim.Adam([translation_params, scale, translation_params1, scale1,
#                              translation_params2, scale2, translation_params3, scale3], lr=1e-2)
    
optimizer = torch.optim.Adam([translation_params, scale], lr=1e-2)
# Run 200 Adam iterations.
import time
t1= time.time()


for t in range(1300):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: apply the mesh operation and render the image.
    translation = translation_params * 1.0
    rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
    
#    translation1 = translation_params1 * 1.0
#    rotation_matrix1 = pyredner.gen_rotate_matrix(euler_angles1)    

#    translation2 = translation_params2 * 1.0
#    rotation_matrix2 = pyredner.gen_rotate_matrix(euler_angles2)  
    
#    translation3 = translation_params3 * 1.0
#    rotation_matrix3 = pyredner.gen_rotate_matrix(euler_angles3)      
    
    if pyredner.get_use_gpu():
        rotation_matrix = rotation_matrix.cuda()
#        rotation_matrix1 = rotation_matrix1.cuda()
#        rotation_matrix2 = rotation_matrix2.cuda()
#        rotation_matrix3 = rotation_matrix3.cuda()
        
    center = torch.mean(torch.cat([shape0_vertices]), 0)
#    center1 = torch.mean(torch.cat([shape1_vertices]), 0)
#    center2 = torch.mean(torch.cat([shape2_vertices]), 0)
#    center3 = torch.mean(torch.cat([shape3_vertices]), 0)
    
    shapes[0].vertices = \
        ((shape0_vertices - center) *scale) @ torch.t(rotation_matrix) + \
        center + translation
#    shapes[1].vertices = \
#        (shape1_vertices - center1)*scale1 @ torch.t(rotation_matrix1) + \
#        center1 + translation1
#
#    shapes[2].vertices = \
#        ((shape2_vertices - center2) *scale2) @ torch.t(rotation_matrix2) + \
#        center2 + translation2
#    shapes[3].vertices = \
#        (shape3_vertices - center3)*scale3 @ torch.t(rotation_matrix3) + \
#        center3 + translation3
        
    shapes[0].normals = pyredner.compute_vertex_normal(shapes[0].vertices, shapes[0].indices)
#    shapes[1].normals = pyredner.compute_vertex_normal(shapes[1].vertices, shapes[1].indices)
#    
#    shapes[2].normals = pyredner.compute_vertex_normal(shapes[2].vertices, shapes[2].indices)
#    shapes[3].normals = pyredner.compute_vertex_normal(shapes[3].vertices, shapes[3].indices)    
    
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4, # We use less samples in the Adam loop.
        max_bounces = 1)
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    img = render(t, *scene_args)
    # Save the intermediate render.
    normalization= torch.tensor([img[:,:,0].max(), img[:,:,1].max(), img[:,:,2].max()], device = pyredner.get_device())
    
    if(t%10==0 or t==1299):
        pyredner.imwrite((img/normalization).cpu(), 'results/'+folder_name+'/iter_{}.png'.format(t))    
#    pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.

#    img_log=  255*torch.log10(img/normalization +1)
    img_log= torch.log2(img/normalization*(2**24)+1)
#    target_log = 255*torch.log10(target +1)
    
    target_log = torch.log2(target*(2**24)+1)
    
#    loss = (img/normalization - target).pow(2).sum()
    loss = (img_log - target_log).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
#    print('translation_params.grad:', translation_params.grad)
#    print('euler_angles.grad:', euler_angles.grad)
#    print('scale.grad:', scale.grad)
#    
#    print('translation_params1.grad:', translation_params1.grad)
#    print('euler_angles1.grad:', euler_angles1.grad)
#    print('scale1.grad:', scale1.grad)
    
#    print('color.grad:', diffuse_reflectance.grad)
    # Take a gradient descent step.
    optimizer.step()
#    # Print the current pose parameters.
#    print('translation:', translation)
#    print('euler_angles:', euler_angles)
#    print('scale:', scale)
#    
#    print('translation1:', translation1)
#    print('euler_angles1:', euler_angles1)
#    print('scale1:', scale1)    
#    print('color/Diffuse Reflectance :', diffuse_reflectance)
    

# Render the final result.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
img = render(1302, *scene_args)
# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/final.exr')
pyredner.imwrite(img.cpu(), 'results/'+folder_name+'/final.png')
normalization= torch.tensor([img[:,:,0].max(), img[:,:,1].max(), img[:,:,2].max()], device = pyredner.get_device())
pyredner.imwrite(torch.abs(target - img/normalization).cpu(), 'results/'+folder_name+'/final_diff.png')

# Convert the intermediate renderings to a video.
#from subprocess import call
#call(["ffmpeg", "-framerate", "24", "-i",
#    "results/"+folder_name+"/iter_%d.png", "-vb", "20M",
#    "results/"+folder_name+"/out.mp4"])

    
print ("Total time taken is {} seconds " .format(time.time() -t1))
#%%
#translation = translation_params * 3.0
#rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
#
#center = torch.mean(torch.cat([shape0_vertices]), 0)
#
#shapes[0].vertices = \
#        ((shape0_vertices - center) *scale) @ torch.t(rotation_matrix) + \
#        center + translation
#
#v_np= torch_to_np(shapes[0].vertices)
#
##Axes3D.plot_surface(X=v_np[:,0], Y=v_np[:,1], Z=v_np[:,2])
#
#x,y,z =v_np[:,0],v_np[:,1], v_np[:,2]; 
#   
#dia=( (x.max()-x.min())+ (y.max()-y.min()) + (z.max()-z.min()))/3
#print('Radius of the Sphere is:', dia/2)
#
#xx,yy,zz= np.ravel(x),np.ravel(y),np.ravel(z)
#
#fig=p.figure()
#ax = p3.Axes3D(fig)
## plot3D requires a 1D array for x, y, and z
## ravel() converts the 100x100 array into a 1x10000 array
#ax.plot3D(xx,yy,zz)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#fig.add_axes(ax)
#p.show()
#
#xc,yc,zc= x.max()- dia/2, y.max()- dia/2, z.max()-dia/2
#print('Centre of the Sphere is:', xc, yc, zc )
#%%    


#v = mesh.vertices/5
#
#v_np= torch_to_np(v)
#
#
#x,y,z =v_np[:,0],v_np[:,1], v_np[:,2];
#
#xx,yy,zz= np.ravel(x),np.ravel(y),np.ravel(z)
#
#fig=p.figure()
#ax = p3.Axes3D(fig)
#ax.plot3D(np.ravel(x),np.ravel(y),np.ravel(z))
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#fig.add_axes(ax)
#p.show()

#%% Scatter Plot for Mesh
#!python
#fig=p.figure()
#ax = p3.Axes3D(fig)
## scatter3D requires a 1D array for x, y, and z
## ravel() converts the 100x100 array into a 1x10000 array
#ax.scatter3D(np.ravel(x),np.ravel(y),np.ravel(z))
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#p.show()

#%%
#translation= torch.tensor([ 0.1040, -7.4208, 31.6828])
#euler_angles= torch.tensor([ 0.1000, -0.1000,  0.1000])
#scale= torch.tensor([3.7331])
#
#
#rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
#    
#center = torch.mean(torch.cat([v]), 0)
#
#v_trans = \
#    ((v - center) *scale) @ torch.t(rotation_matrix) + \
#    center + translation
#
#v_np= torch_to_np(v_trans)
#
##Axes3D.plot_surface(X=v_np[:,0], Y=v_np[:,1], Z=v_np[:,2])
#
#x,y,z =v_np[:,0],v_np[:,1], v_np[:,2];
#
#xx,yy,zz= np.ravel(x),np.ravel(y),np.ravel(z)
#
#fig=p.figure()
#ax = p3.Axes3D(fig)
## plot3D requires a 1D array for x, y, and z
## ravel() converts the 100x100 array into a 1x10000 array
#ax.plot3D(x,y,z)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#fig.add_axes(ax)
#p.show()

    
    
    
    