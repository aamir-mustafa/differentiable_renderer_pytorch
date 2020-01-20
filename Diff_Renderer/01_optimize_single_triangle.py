# The Python interface of redner is defined in the pyredner package
import pyredner
import torch



pyredner.set_use_gpu(torch.cuda.is_available())
#pyredner.set_use_gpu(False)

#First, we setup a camera, by constructing a pyredner.Camera object 

# I think this is known and not optimized

cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)



#Next, we setup the materials for the scene. All materials in the scene are stored in a single Python list.
#The index of a material in the list is its material id. Our simple scene only has a single grey 
#material with reflectance 0.5 

# This material gives color to the initial guess as well as the target image.

# [0.5, 0.5, 0.5] is the color for gray
# Lets try green with [0,255,0] (has to be floats as [0.0, 1.0, 0.0] )

#mat_grey = pyredner.Material(\
#    diffuse_reflectance = \
#        torch.tensor([0.5, 0.5, 0.5], device = pyredner.get_device()))

mat_grey = pyredner.Material(\
    diffuse_reflectance = \
        torch.tensor([1.0, 0.0, 0.0], device = pyredner.get_device()))

# The material list of the scene # 
materials = [mat_grey]

# Shape of the target image

#shape_triangle = pyredner.Shape(\
#    vertices = torch.tensor([[-1.7, 1.0, 0.0], [1.0, 1.0, 0.0], [-0.5, -1.0, 0.0]],
#        device = pyredner.get_device()),
#    indices = torch.tensor([[0, 1, 2]], dtype = torch.int32,
#        device = pyredner.get_device()),
#    uvs = None,
#    normals = None,
#    material_id = 0)

#[0, 1, 2] are the right, left and bottom indices of the traingle

# Here the z axis for the traingle is 0 for all three vertices [x, y, 0]
# Some portion of the traingle may get cropped due to the relative positioning of the light source and the traingle

# This target image is also first rendered and then the initial guess is made close to it
#(Can we have a png intead of the rendering paramenters of the target img and then optimize our initial guess
#to generate a final image?)

shape_triangle = pyredner.Shape(\
    vertices = torch.tensor([[-2.7, 1.0, 0.0], [1.0, 2.0, 0.0], [-0.5, -1.0, 0.0]],
        device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2]], dtype = torch.int32,
        device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)
    
# Merely having a single triangle is not enough for physically-based rendering. We need to have a light source.
# Here we setup the shape of a quad area light source, similary to the previous triangle.
    
    # What is a quad area light
    #An Area Light is defined by a rectangle in space. Light is emitted in all directions uniformly across 
    #their surface area, but only from one side of the rectangle. There is no manual control for the range 
    #of an Area Light, however intensity will diminish at inverse square of the distance as it travels away 
    #from the source. Since the lighting calculation is quite processor-intensive, area lights are not available
    #at runtime and can only be baked into lightmaps

# Again z axis for the fixed for the rectangle area light source    
#shape_light = pyredner.Shape(\
#    vertices = torch.tensor([[-1.0, -1.0, -7.0],
#                             [ 1.0, -1.0, -7.0],
#                             [-1.0,  1.0, -7.0],
#                             [ 1.0,  1.0, -7.0]], device = pyredner.get_device()),
#    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
#        dtype = torch.int32, device = pyredner.get_device()),
#    uvs = None,
#    normals = None,
#    material_id = 0)


# Changing this doesnt change anything, but makes the target image brighter ot lighter
    # Since the light source is common between the target and the init image, even the init image 
    # becomes brighter
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
shapes = [shape_triangle, shape_light]


light = pyredner.AreaLight(shape_id = 1, 
                           intensity = torch.tensor([20.0,20.0,20.0]))
area_lights = [light]
# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam, shapes, materials, area_lights)


#All PyTorch functions take a flat array of PyTorch tensors as input, therefore we 
#need to serialize the scene into an array. The following function does this. We also
# specify how many Monte Carlo samples we want to use per pixel and the number of bounces 
#for indirect illumination here (one bounce means only direct illumination).

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

#Now we render the scene as our target image. To render the scene, we use our custom PyTorch 
#function in pyredner/render_pytorch.py . First setup the alias of the render function

render = pyredner.RenderFunction.apply

img = render(0, *scene_args)     # img is the target image # img.shape is torch.Size([256, 256, 3])

#This generates a PyTorch tensor with size [width, height, 3]. The output image is in the 
#GPU memory if you are using GPU. Now we save the generated image to disk.

pyredner.imwrite(img.cpu(), 'results/optimize_single_triangle/target.exr')
pyredner.imwrite(img.cpu(), 'results/optimize_single_triangle/target.png')
#%%
#Now we read back the target image we just saved, and copy to GPU if necessary:
target = pyredner.imread('results/optimize_single_triangle/target.exr') # target.shape is torch.Size([256, 256, 3])
#pyredner.imread also reads png images as the same
if pyredner.get_use_gpu():
    target = target.cuda()

#Next we want to produce the initial guess. We do this by perturb the scene.
#  Here I will check if we can load any image of a triangle and optimize our guess to the shape of the
#target img
     
shape_triangle.vertices = torch.tensor(\
    [[-2.0,1.5,0.3], [0.9,1.2,-0.3], [-0.4,-1.4,0.2]],
    device = pyredner.get_device(),
    requires_grad = True)  # Becasue these values have to be optimized ie why  requires_grad = True

#%% from here on to the loop starts, this is only to print/show the init guess image
# It has nothing to do with the actual rendering of the init img or the optimization
# ie we run the scene_args again inisde the loop and render the image

#We need to serialize the scene again to get the new arguments. We then render our initial guess
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

img = render(1, *scene_args)   #img is the init guess image # img.shape is torch.Size([256, 256, 3])

# render takes a number as first argument 
#Check what this number is? Becasue in the final case we give 202 as the final argument to the render

# Save the image
pyredner.imwrite(img.cpu(), 'results/optimize_single_triangle/init.png')
# Compute the difference and save the images.
diff = torch.abs(target - img)   # difference between the rgb values
pyredner.imwrite(diff.cpu(), 'results/optimize_single_triangle/init_diff.png')


#%%
#Now we want to refine the initial guess using gradient-based optimization. 
#We use PyTorch's optimizer to do this.
optimizer = torch.optim.Adam([shape_triangle.vertices], lr=5e-2) # what is to be optimized


for t in range(200):
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
    pyredner.imwrite(img.cpu(), 'results/optimize_single_triangle/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()   # simple l2 loss between the rgb values of the image
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients of the three vertices.
    print('grad:', shape_triangle.vertices.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current three vertices.
    print('vertices:', shape_triangle.vertices)

#%%
# Render the final result.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = render(202, *scene_args)
# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/optimize_single_triangle/final.exr')
pyredner.imwrite(img.cpu(), 'results/optimize_single_triangle/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/optimize_single_triangle/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/optimize_single_triangle/iter_%d.png", "-vb", "20M",
    "results/optimize_single_triangle/out.mp4"])
