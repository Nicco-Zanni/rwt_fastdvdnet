import numpy as np
from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt
import scipy.io as sio
#for noise
from PIL import Image
from scipy.stats import ks_2samp
import argparse
import os
import torch


def inv_sampling(x,bin_width= 0.00001,test_samples=500):        
        dist_x,mids = dist(x,bin_width)
        cumulative = np.cumsum(dist_x)
        cumulative = cumulative - np.amin(cumulative)
        f = interp1d(cumulative/np.amax(cumulative), mids)
        #plt.hist(x.flatten(),label=['original'])
        #plt.hist(f(np.random.random(test_samples)).flatten(),alpha=0.4, label=['sampled'])
        #plt.legend(loc='upper right')
        #plt.show()
        #f(np.random.random(pnts))
        return f

def dist(x,bin_width = 0.00001 ):
    hist, bin_edges = np.histogram(x, bins=np.linspace(np.amin(x),np.amax(x),int((np.amax(x)-np.amin(x))/bin_width)))
    hist = hist / x.size
    mids = bin_edges[:-1] + np.diff(bin_edges)/2
    return hist,mids

def load_param(root_path='./'):
    matdir_b = os.path.join(root_path, 'dataset_var_samples_q.mat')
    mat_b = sio.loadmat(matdir_b)
    intrcpt_R = mat_b['b_array_R_d']
    intrcpt_G = mat_b['b_array_G_d']
    intrcpt_B = mat_b['b_array_B_d']

    matdir_a_slope_R = os.path.join(root_path, 'full_dataset_parameter_R.mat')
    matdir_a_slope_G = os.path.join(root_path, 'full_dataset_parameter_G.mat')
    matdir_a_slope_B = os.path.join(root_path, 'full_dataset_parameter_B.mat')
    #R parameters
    mat_a_slope_R = sio.loadmat(matdir_a_slope_R)
    a_R = mat_a_slope_R['a_array_R_d']
    m_R = mat_a_slope_R['slope_array_R']
    a_R = a_R[a_R>0]
    #G parameters
    mat_a_slope_G = sio.loadmat(matdir_a_slope_G)
    a_G = mat_a_slope_G['a_array_G_d']
    m_G = mat_a_slope_G['slope_array_G']
    a_G = a_G[a_G>0]
    #B parameters
    mat_a_slope_B = sio.loadmat(matdir_a_slope_B)
    a_B = mat_a_slope_B['a_array_B_d']
    m_B = mat_a_slope_B['slope_array_B']
    a_B = a_B[a_B>0]

    f_intrcpt_R = inv_sampling(intrcpt_R,test_samples=intrcpt_R.size)
    f_intrcpt_G = inv_sampling(intrcpt_G,test_samples=intrcpt_G.size)
    f_intrcpt_B = inv_sampling(intrcpt_B,test_samples=intrcpt_B.size)

    f_m_R = inv_sampling(m_R,test_samples=m_R.size)
    f_m_G = inv_sampling(m_G,test_samples=m_G.size)
    f_m_B = inv_sampling(m_B,test_samples=m_B.size)

    f_a_R = inv_sampling(a_R,test_samples=a_R.size)
    f_a_G = inv_sampling(a_G,test_samples=a_G.size)
    f_a_B = inv_sampling(a_B,test_samples=a_B.size)

    return f_intrcpt_R,f_m_R,f_a_R,f_intrcpt_G,f_m_G,f_a_G,f_intrcpt_B,f_m_B,f_a_B

def sample_param(f_intercept,f_slope,f_a,n_samples=1):
    
    intercept = f_intercept(np.random.random(n_samples))
    slope     = f_slope(np.random.random(n_samples))
    a         = f_a(np.random.random(n_samples))
    b         = slope*a+intercept
    return a,b

def sample_param_RGB(f_intrcpt_R,f_m_R,f_a_R,f_intrcpt_G,f_m_G,f_a_G,f_intrcpt_B,f_m_B,f_a_B):
    repeat = True
    while repeat:
        a_R,b_R = sample_param(f_intrcpt_R,f_m_R,f_a_R)
        if b_R>0:
            repeat = False
    
    repeat = True
    while repeat:
        a_G,b_G = sample_param(f_intrcpt_G,f_m_G,f_a_G)
        if b_G>0:
            repeat = False

    repeat = True
    while repeat:
        a_B,b_B = sample_param(f_intrcpt_B,f_m_B,f_a_B)
        if b_B>0:
            repeat = False

    a = np.array([a_R[0],a_G[0],a_B[0]])
    b = np.array([b_R[0],b_G[0],b_B[0]])
    return a,b

def add_noise(img, a_array, b_array):
    #print(img.shape)
    img_dim = img.ndim
    if img_dim==2:
        ch_n = 1.0
        print("Warning: Code is for RGB noise")
    else:
        ch_n = img.shape[2]
    z= np.zeros(img.shape)
    for i in np.arange(0,ch_n):
        y = img[:,:,i]
        a = a_array[i]
        b = b_array[i]
        if a==0:   # no Poissonian component
            z_i=y
        else:      #% Poissonian component
            chi=1./a
            z_i = np.random.poisson(np.maximum(0,chi*y))/chi
        pois = z_i
        z_i=z_i+np.sqrt(np.maximum(0,b))*np.random.normal(loc=0.0, scale=1.0, size=y.shape)  #% Gaussian component
        z[:,:,i] = z_i
    #clipping
    z = np.clip(z, 0.0, 1.0)
    return z


def add_noise_tensor(img, a, b):
    """
    Vectorized noise addition for a PyTorch tensor.
    Args:
        img (torch.Tensor): Input tensor of shape [N*F, C, H, W] (normalized to [0, 1]).
        a (torch.Tensor): Tensor of shape [N*F, C], Poisson noise parameters per channel.
        b (torch.Tensor): Tensor of shape [N*F, C], Gaussian noise parameters per channel.
    Returns:
        torch.Tensor: Noisy tensor of shape [N*F, C, H, W].
    """
    chi = 1.0 / a.unsqueeze(-1).unsqueeze(-1)  # Shape: train [N*F, C, 1, 1] , val [F, C, 1, 1]
    poisson_noise = torch.poisson(torch.clamp(chi * img, min=0)) / chi
    gaussian_noise = torch.randn_like(img) * torch.sqrt(torch.clamp(b.unsqueeze(-1).unsqueeze(-1), min=0))
    noisy_tensor = poisson_noise + gaussian_noise
    return torch.clamp(noisy_tensor, min=0, max=1)


def generate_train_noisy_tensor(img, noise_gen_folder, device='cuda'):
    """
    Add noise to a tensor of shape [N, F, C, H, W].
    Args:
        img (torch.Tensor): Input tensor of shape [N, F, C, H, W] (not normalized).
        noise_gen_folder (str): Path to noise parameters.
    Returns:
        torch.Tensor: Noisy tensor of shape [N, F, C, H, W].
    """
    # Load noise parameters
    f_intrcpt_R, f_m_R, f_a_R, f_intrcpt_G, f_m_G, f_a_G, f_intrcpt_B, f_m_B, f_a_B = load_param(noise_gen_folder)

    # Reshape to [N*F, C, H, W]
    N, F, C, H, W = img.shape
    img_tensor = img.view(-1, C, H, W) / 255.0  # [N*F, C, H, W] normalized

    # Sample noise parameters for each frame
    a_array = torch.zeros((N * F, C), device=device)
    b_array = torch.zeros((N * F, C), device=device)

    for c, (f_intrcpt, f_m, f_a) in enumerate(zip([f_intrcpt_R, f_intrcpt_G, f_intrcpt_B],
                                                  [f_m_R, f_m_G, f_m_B],
                                                  [f_a_R, f_a_G, f_a_B])):
        for f in range(N * F):
            repeat = True
            while repeat:
                a, b = sample_param(f_intrcpt, f_m, f_a, n_samples=1)
                repeat = False if b > 0 else True
            a_array[f, c] = torch.tensor(a, device=device)
            b_array[f, c] = torch.tensor(b, device=device)
    
    # Apply noise
    noisy_tensor = add_noise_tensor(img_tensor, a_array, b_array)

    # Reshape back to [N, F, C, H, W]
    noisy_tensor = noisy_tensor.view(N, F, C, H, W)

    return (noisy_tensor * 255).round().clamp(0, 255)


def generate_val_noisy_tensor(seq, noise_gen_folder, device='cuda'):
    """
    Add noise to a tensor of shape [F, C, H, W].
    Args:
        seq (torch.Tensor): Input tensor of shape [F, C, H, W] (not normalized).
        noise_gen_folder (str): Path to noise parameters.
    Returns:
        torch.Tensor: Noisy tensor of shape [N, F, C, H, W].
    """
    # Load noise parameters
    f_intrcpt_R, f_m_R, f_a_R, f_intrcpt_G, f_m_G, f_a_G, f_intrcpt_B, f_m_B, f_a_B = load_param(noise_gen_folder)

    # Reshape to [N*F, C, H, W]
    F, C, H, W = seq.shape
    seq = seq / 255.0  # [N*F, C, H, W] normalized

    # Sample noise parameters for each frame
    a_array = torch.zeros((F, C), device=device)
    b_array = torch.zeros((F, C), device=device)

    for c, (f_intrcpt, f_m, f_a) in enumerate(zip([f_intrcpt_R, f_intrcpt_G, f_intrcpt_B],
                                                  [f_m_R, f_m_G, f_m_B],
                                                  [f_a_R, f_a_G, f_a_B])):
        for f in range(F):
            repeat = True
            while repeat:
                a, b = sample_param(f_intrcpt, f_m, f_a, n_samples=1)
                repeat = False if b > 0 else True
            a_array[f, c] = torch.tensor(a, device=device)
            b_array[f, c] = torch.tensor(b, device=device)
    
    # Apply noise
    noisy_seq = add_noise_tensor(seq, a_array, b_array)

    return (noisy_seq * 255).round().clamp(0, 255)

 
def to_ImageFromArray(a):
       return Image.fromarray((a * 255.0).round().clip(0, 255).astype(np.uint8))
       

def noise_sampling(gt_dir, n_obs, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gt_images = [img for img in os.listdir(gt_dir) if img.endswith(('.jpg', '.png', '.tiff'))]
    for image in gt_images:
        print("Processing image: {}/{}".format(gt_dir, image))
        img_gt = np.asarray(Image.open(os.path.join(gt_dir, image)),dtype=np.float64)/255.0  # [H, W, C]
        n_obs = n_obs
        f_intrcpt_R,f_m_R,f_a_R,f_intrcpt_G,f_m_G,f_a_G,f_intrcpt_B,f_m_B,f_a_B = load_param()

        for i in np.arange(n_obs):
            #sample noise parameters
            a,b = sample_param_RGB(f_intrcpt_R,f_m_R,f_a_R,f_intrcpt_G,f_m_G,f_a_G,f_intrcpt_B,f_m_B,f_a_B)
            #print(a,b)
            #add noise to image and clip
            img_syn_noisy = add_noise(img_gt,a,b) 
            #scale and qunatize
            img_syn_noisy_q = (img_syn_noisy * 255.0).round().clip(0, 255).astype(np.uint8)
            #save
            img_name = " ".join(image.split(".")[:-1]) 
            output_name = output_dir+"/"+img_name+"_n"+str(i+1)+".png"
            #show/save       
            to_ImageFromArray(img_syn_noisy).save(output_name)   
            #to_ImageFromArray(img_syn_noisy).show()


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument("--input_path", "-i", help="Directory of input images")
    parser.add_argument("--n_obs", "-n", default=1, type=int, help="Number of generated noisy image for each image, Default=1")
    parser.add_argument("--output_path", "-o", help="Directory of output images")
    parser.add_argument("--multiple", "-m", action='store_true', help="Generate noise for multiple image sequences")
    args = parser.parse_args()
    
    if args.multiple:
        for folder in os.listdir(args.input_path):
            noise_sampling(args.input_path+"/"+folder, args.n_obs, args.output_path+"/"+folder)
    else:
        noise_sampling(args.input_path, args.n_obs, args.output_path)
        
