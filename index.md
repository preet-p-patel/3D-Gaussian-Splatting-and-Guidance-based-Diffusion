# Learning for 3D Vision - Assignment 4

## 1. 3D Gaussian Splatting

### 1.1 3D Guassian Rasterization 

<img src="Q1\output\q1_1_render.gif" alt="1.1" >

### 1.2 Training 3D Gaussian Representations

Learning Rates:

<table>
    <tr>
        <th>Opacities</th>
        <th>Scales</th>
        <th>Colours</th>
        <th>Means</th>
    </tr>
    <tr>
        <th>0.005</th>
        <th>0.01</th>
        <th>0.02</th>
        <th>0.001</th>   
    </tr>
</table>

Number of iterations: 250

PSNR: 28.470

SSIM: 0.923


<table>
    <tr>
        <th>Training Progress</th>
    </tr>
    <tr>
        <td><img src="Q1\output\q1_training_progress.gif" alt="Grid Visualization"></td>    
    </tr>
</table>


<table>
    <tr>
        <th>Final Output</th>
    </tr>
    <tr>
        <td><img src="Q1\output\q1_training_final_renders.gif" alt="Grid Visualization" width= 400></td>    
    </tr>
</table>

## 1.3 Extensions 

### 1.3.1 Rendering with Spherical Harmoics

<table>
    <tr>
        <th>With Spherical Harmoincs</th>
        <td><img src="Q1\output\q1_3_render_spherical_harmoics.gif" alt="Grid Visualization"></td>
    </tr>
    <tr>
        <th>Without Spherical Harmonics</th>
        <td><img src="Q1\output\q1_1_render.gif" alt="Grid Visualization"></td>   
    </tr>
</table>


Here the 2nd output in each case shows output with shperical harmonics, we can see proper variation of colors and shading based on the light and object orientation in these cases unlike the 1 output


<table>
    <tr>
        <th>Frame</th>
        <th></th>
    </tr>
    <tr>
        <th>5</th>
        <td><img src="Q1\output\q1_1_render\005.png" alt="Grid Visualization">
        <img src="Q1\output\q1_3_render_spherical_harmonics\005.png" alt="Grid Visualization"></td>   
    </tr>
    <tr>
        <th>10</th>
        <td><img src="Q1\output\q1_1_render\010.png" alt="Grid Visualization">
        <img src="Q1\output\q1_3_render_spherical_harmonics\010.png" alt="Grid Visualization"></td>   
    </tr>
    <tr>
        <th>15</th>
        <td><img src="Q1\output\q1_1_render\015.png" alt="Grid Visualization">
        <img src="Q1\output\q1_3_render_spherical_harmonics\015.png" alt="Grid Visualization"></td>   
    </tr>
</table>



### 1.3.2 Training on Harder Scene

<table>
    <tr>
        <th>Attempt</th>
        <th>Outputs</th>
    </tr>
    <tr>
        <th>Baseline (Trained with same configuration as before)</th>
        <td><img src="Q1\output\attempt1_q1_harder_training_final_renders.gif" alt="Grid Visualization">
        <img src="Q1\output\attemtp1_q1_harder_training_progress.gif" alt="Grid Visualization"></td>   
    </tr>
    <tr>
        <th>Attempt 2 (Tweaking learning rates and training for longer)</th>
        <td><img src="Q1\output\q1_3_attempt2_harder_training_final_renders.gif" alt="Grid Visualization">
        <img src="Q1\output\q1_3_attempt2_harder_training_progress.gif" alt="Grid Visualization"></td>   
    </tr>
    
</table>

## 2. Diffusion-guided Optimization

### 2.1 SDS Loss + Image Optimization

<table>
    <tr>
        <th>Prompt</th>
        <th>No guidance</th>
        <th>with Guidance</th>
    </tr>
    <tr>
        <th>A hamburger (2000 iterations)</th>
        <td><img src="Q2\output\image\A_hamburger_no_guidance\output.png" alt="Grid Visualization" width = 250></td> 
        <td><img src="Q2\output\image\A_hamburger_guidance\output.png" alt="Grid Visualization" width = 250></td>   
    </tr>
    <tr>
        <th>A standing corgi dog (2000 iterations)</th>
        <td><img src="Q2\output\image\A_standing_corgi_dog_no_guidance\output.png" alt="Grid Visualization" width = 250></td> 
        <td><img src="Q2\output\image\A_standing_corgi_dog_guidance\output.png" alt="Grid Visualization" width = 250></td>   
    </tr>
    <tr>
        <th>A beach house (2000 iterations)</th>
        <td><img src="Q2\output\image\A_beach_housebeach_house_no_guidance\output.png" alt="Grid Visualization" width = 250></td> 
        <td><img src="Q2\output\image\A_beach_housebeach_house_guidance\output.png" alt="Grid Visualization"width = 250></td>   
    </tr>
    <tr>
        <th>Pittsburgh Downtown (2000 iterations)</th>
        <td><img src="Q2\output\image\pittsburgh_downtownPittsburgh_downtown_no_guidance\output.png" alt="Grid Visualization" width = 250></td> 
        <td><img src="Q2\output\image\pittsburgh_downtownPittsburgh_downtown_guidance\output.png" alt="Grid Visualization" width = 250></td>   
    </tr>
</table>

### 2.2 Texture Map Optimization for Mesh

<table>
    <tr>
        <th>A cow with spots</th>
        <th>A cow with black and white spots</th>
    </tr>
    <tr>
        <td><img src="Q2\output\mesh\A_cow_with_spotscow_with_spots\final_mesh.gif" alt="Grid Visualization" width = 250></td> 
        <td><img src="Q2\output\mesh\cow_with_black_and_white_spots\final_mesh.gif" alt="Grid Visualization" width = 250></td>   
    </tr>
</table>

### 2.3 NeRF Optimization

<table>
    <tr>
        <th>Prompt</th>
        <th>Rendered RGB</th>
        <th>Depth</th>
    </tr>
    <tr>
        <th>A standing corgi</th>
        <td><video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_corgi_dog_no_view_attempt2\videos\rgb_ep_60.mp4" type="video/mp4">
</video></td> 
        <td><video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_corgi_dog_no_view_attempt2\videos\depth_ep_60.mp4" type="video/mp4">
</video></td>   
    </tr>
    <tr>
        <th>A flying dragon</th>
        <td><video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_flying_dragonno_view\videos\rgb_ep_40.mp4" type="video/mp4">
</video></td> 
        <td><video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_flying_dragonno_view\videos\depth_ep_40.mp4" type="video/mp4">
</video></td>   
    </tr>
    <tr>
        <th>A flower pot</th>
        <td><video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_flower_pot_no_view\videos\rgb_ep_20.mp4" type="video/mp4">
</video></td> 
        <td><video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_flower_pot_no_view\videos\depth_ep_20.mp4" type="video/mp4">
</video></td>   
    </tr>
</table>

## 2.4 Extensions

### 2.4.1 View Dependent Text Embedding

*Effect of view dependent text embedding:* We can see clearly in the corgi prompt that without view dependence it has weird features like multiple faces but with view dependence it is able to create better features which are more realistic.

<table>
    <tr>
        <th>Prompt</th>
        <th>With View Dependence</th>
        <th>Without View Dependence</th>
    </tr>
    <tr>
        <th>A standing corgi dog</th>
        <td><video width="250" height="250" controls>
  <source src="Q2\output\nerf\Standing_corgi_view\videos\rgb_ep_49.mp4" type="video/mp4">
</video> <video width="250" height="250" controls>
  <source src="Q2\output\nerf\Standing_corgi_view\videos\depth_ep_49.mp4" type="video/mp4">
</video>
        <td><video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_corgi_dog_no_view_attempt2\videos\rgb_ep_60.mp4" type="video/mp4">
</video> <video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_corgi_dog_no_view_attempt2\videos\depth_ep_60.mp4" type="video/mp4">
</video>
        <td>   
    </tr>
    <tr>
        <th>A flying dragon</th>
        <td><video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_flying_dragon_view\videos\rgb_ep_49.mp4" type="video/mp4">
</video> <video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_flying_dragon_view\videos\depth_ep_49.mp4" type="video/mp4">
</video> 
        <td><video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_flying_dragonno_view\videos\rgb_ep_40.mp4" type="video/mp4">
</video> <video width="250" height="250" controls>
  <source src="Q2\output\nerf\a_flying_dragonno_view\videos\depth_ep_40.mp4" type="video/mp4">
</video>
        <td>   
    </tr>
    
</table>
