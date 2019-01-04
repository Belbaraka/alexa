import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from tqdm import tqdm

#CONSTANTS

fs = 8000 # sampling frequencey
m = 50 # number of sources in training stage
M = 49 # number of sources in test stage
L = 5 # number of perturbations around a given source 
d = 2 # degrees of freedom (in this case azimuth and elevation angle)
epsilon = 1e20 # scaling factor
eps_var = 1e-5 # sclaing factor
k = 3 # number of neigherest neighbour selected


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan(y/x)
    
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return x, y, z

def feature_Vector_src(index_src, D=256):
    '''
    This function creates the feature vector for given source/perturbation position using the microphones in the room
    '''
    
    overlap = 1
    rir0 = room.rir[0][index_src]
    rir1 = room.rir[1][index_src]
    len0, len1 = rir0.shape[0], rir1.shape[0]

    atf_0 = pra.stft(rir0, L=D, hop=int(len0 * overlap), win=pra.windows.hann(N=256))
    atf_1 = pra.stft(rir1, L=D, hop=int(len1 * overlap), win=pra.windows.hann(N=256))
    
    return atf_0 / atf_1

def feature_Vector_perturb(index_perturb, D=256):
    '''
    This function creates the feature vector for given source/perturbation position using the microphones in the room
    '''
    
    overlap = 1
    rir0 = room_perturb.rir[0][index_perturb]
    rir1 = room_perturb.rir[1][index_perturb]
    len0, len1 = rir0.shape[0], rir1.shape[0]

    atf_0 = pra.stft(rir0, L=D, hop=int(len0 * overlap), win=pra.windows.hann(N=256))
    atf_1 = pra.stft(rir1, L=D, hop=int(len1 * overlap), win=pra.windows.hann(N=256))
    
    return atf_0 / atf_1


def feature_Vector_testPos(index_src, D=256):
    '''
    This function creates the feature vector for given test position using the microphones in the room
    '''
    
    overlap = 1
    rir0 = room_test.rir[0][index_src]
    rir1 = room_test.rir[1][index_src]
    len0, len1 = rir0.shape[0], rir1.shape[0]

    atf_0 = pra.stft(rir0, L=D, hop=int(len0 * overlap), win=pra.windows.hann(N=256))
    atf_1 = pra.stft(rir1, L=D, hop=int(len1 * overlap), win=pra.windows.hann(N=256))
    
    return atf_0 / atf_1

def create_training(mic, m=481, azimuth=[math.pi/16, 2*math.pi/16], elevation=[7*math.pi/16, math.pi/2], radius = 1):
    '''
    This function generates m training samples located on a sector of a sphere around the first microphone. This sphere has a fixed radius of 1 m
    '''
    training_Set_xyz = []
    training_Set_spherical = []
    epsilon = 1e-7
    for i in range(m):
        azimuth_i = np.random.uniform(low=azimuth[0], high=azimuth[1]+epsilon)
        elevation_i = np.random.uniform(low=elevation[0], high=elevation[1]+epsilon)
        pos_xyz_i = mic + spherical_to_cartesian(radius, elevation_i, azimuth_i)
        training_Set_xyz.append(pos_xyz_i)
        training_Set_spherical.append((radius, elevation_i, azimuth_i))
    return training_Set_xyz, training_Set_spherical


def create_perturbations(training_Set_sph, mic, L=20):
    '''
    This function generates the L different perturbations around the source position
    '''
    nb_src = len(training_Set_sph)
    postions_Of_perturbations = []
    
    for src_pos in training_Set_sph:
        for i in range(L):
            perturbations_i = np.random.randn(2)
            pos_xyz_perturb_i = mic + spherical_to_cartesian(src_pos[0], src_pos[1] + perturbations_i[0], src_pos[2] + perturbations_i[1])
            postions_Of_perturbations.append(pos_xyz_perturb_i)
        
    return postions_Of_perturbations

def covMatrix_from_perturb(feature_vecs_set, L=20, D=256):
    
    cov = np.zeros((D,D), dtype='complex128')
    for feat_vec in feature_vecs_set:
        cov += feat_vec * feat_vec.T
    
    return cov/L


def index_cov_matrix(index_k, index_l, training_positions):
    if index_k == index_l:
        return index_k
    else:
        pos_k, pos_l = training_positions[index_k], training_positions[index_l]
        pos_middle = 0.5*(pos_k + pos_l)
        current_dist_min = np.linalg.norm(pos_middle - pos_k)
        index_min = index_k
        for i,pos in enumerate(training_positions):
            dist = np.linalg.norm(pos_middle - pos)
            if dist < current_dist_min:
                current_dist_min = dist
                index_min = i
        return index_min
    

    
def normalizing_factor(index_k, index_l, training_positions, cov_matrices):
    index = index_cov_matrix(index_k, index_l, training_positions)
    return np.sqrt(np.linalg.det(cov_matrices[index]))


def compute_elem_W_kl(index_k, index_l, cov_matrices, training_positions, epsilon = epsilon):
    #How to choose epsilon ?
    feature_vec_k, feature_vec_l = feature_Vector_src(index_k), feature_Vector_src(index_l)
    inv_cov_sum = np.linalg.inv(cov_matrices[index_k] + cov_matrices[index_l])
    
    nom = (feature_vec_k - feature_vec_l) @ inv_cov_sum @ (feature_vec_k - feature_vec_l).T
    #d_kl = normalizing_factor(index_k, index_l, training_positions, cov_matrices)
    #print((np.pi/d_kl) * np.exp(-nom/epsilon))
    d_kl = 1
    return (np.pi/d_kl) * np.exp(-nom/epsilon)


def compute_affinity_matrix(cov_matrices, training_positions, m=m):
    W = np.zeros((m, m), dtype='complex128')
    for k in tqdm(range(m)):
        for l in range(m):
            W[k, l] = compute_elem_W_kl(k, l, cov_matrices, training_positions)
            
    return W

def compute_feature_Vecs_set(src_index, postions_Of_perturbations, L=L):
    '''
    This function compute the feature vectors of the perturbations around the source with index `src_index`
    '''
    feature_vec_set = []
    for i,_ in enumerate(postions_Of_perturbations[src_index:L+src_index]):
        index_perturb = src_index*L + i
        feature_vec_set.append(feature_Vector_perturb(index_perturb))
    return feature_vec_set

def compute_covMatrices(postions_Of_perturbations, m=m, L=L):
    '''
    This function computes the covariance matrices of the sources in the training set
    '''
    cov_matrices = []
    for src_index in tqdm(range(m)):
        feature_vecs_set = compute_feature_Vecs_set(src_index, postions_Of_perturbations, L=L)
        cov_matrices.append(covMatrix_from_perturb(feature_vecs_set))
    
    return cov_matrices





def left_singular_vect(index_j, eig_val_j, eig_vect_j, A_tilde):
    return ((1/np.sqrt(eig_val_j))*A_tilde) @ eig_vect_j

def compute_d_left_sing_vect(len_, eig_vals, eig_vects, A_tilde, d=d):
    '''
    This function computes the lefts singular vectors of A_tilde as the weighted interpolation of the 
    eigenvectors of W
    '''
    singular_vects = np.zeros((len_, d), dtype='complex128')
    for i in range(d):
        singular_vects[:, i] = left_singular_vect(i, eig_vals[i], eig_vects[i], A_tilde)
        
    return singular_vects

def get_embedding(index_feat_vect, singular_vects):
    '''
    This function reconstructs an embedding of the measurements onto the space spanned by the d left singular vectors
    '''
    return singular_vects[index_feat_vect, :]


def interpolation_coef(index_j, index_test_pos, singular_vects, train_featVects, indices_knn, eps_var=eps_var):
    '''
    This function comptes the interpolation coefficient used to estimate the test positions
    '''
    embedding_T_i = singular_vects[m + index_test_pos, :]
    embedding_T_j = singular_vects[index_j, :]
    
    num = np.exp(-np.linalg.norm(embedding_T_i - embedding_T_j)**2/eps_var)
    
    denum = 0
    for s in indices_knn:
        embedding_T_s = singular_vects[s, :]
        denum += np.exp(-np.linalg.norm(embedding_T_i - embedding_T_s)**2/eps_var)
    return num / denum


def estimate_theta_i(index_test_pos, singular_vects, train_featVects, training_Set_spherical, eps_var=eps_var):
    '''
    This function estimates the test position
    '''
    theta_i = np.zeros((2,))
    indices_knn = k_nearest_featVect(feature_Vector_testPos(index_src=index_test_pos), train_featVects, k=k)
    for j in indices_knn:
        theta_j = np.array([training_Set_spherical[j][1], training_Set_spherical[j][2]])
        theta_i += theta_j * interpolation_coef(j, index_test_pos, singular_vects, train_featVects, indices_knn, eps_var)
        
    return theta_i


def compute_train_featVects(m=m):
    train_featVects = []
    for i in range(m):
        train_featVect = feature_Vector_src(index_src=i)
        train_featVects.append(train_featVect)
    return train_featVects


def k_nearest_featVect(test_featVect, train_featVects, k=k):
    
    m = len(train_featVects)
    distances = []
    for i in range(m):
        dist = np.linalg.norm(test_featVect - train_featVects[i])
        distances.append(-dist)

    return np.array(distances).argsort()[-k:][::-1]

def estimation_error(theta_gt, theta_pred):
    num = np.linalg.norm(theta_gt - theta_pred)
    denum = np.linalg.norm(theta_gt)
    return num / denum

def rmse(thetas_gt, thetas_pred):
    M = len(thetas_gt)
    rmse = 0
    for i in range(M):
        rmse += estimation_error(thetas_gt[i], thetas_pred[i])**2
    
    rmse = np.sqrt(rmse/M)
    return rmse

##TRAINING STAGE

#Create sources positions and their respective perturbated positions

# Add 2 microphones
mics = np.array([[3, 1, 1], [3.2, 1, 1]])
mic_array = pra.MicrophoneArray(mics.T, fs)
mic0_pos = mic_array.R[:,0]
mic1_pos = mic_array.R[:,1]

training_Set_xyz, training_Set_spherical = create_training(mic0_pos, m=m)
postions_Of_perturbations = create_perturbations(training_Set_spherical, mic0_pos, L=L)


#Create room where training sources are added
# Create a 6.5 by 3.5  by 2 metres shoe box room
room = pra.ShoeBox([6.5, 3.5, 2], fs=fs, max_order=3)

room.add_microphone_array(mic_array)

# Add wgn of 3 seconds duration 
duration_samples = 3*int(fs) 

for pos in training_Set_xyz:
    source_signal = np.random.randn(duration_samples)
    room.add_source(pos, signal=source_signal)

room.image_source_model(use_libroom=False)
room.compute_rir()

# Create the exact same room, where we wil add only the perturbed postions 
room_perturb = pra.ShoeBox([6.5, 3.5, 2], fs=fs, max_order=3)
room_perturb.add_microphone_array(mic_array)

# Add wgn of 3 seconds duration 
duration_samples = 3*int(fs) 

for pos in postions_Of_perturbations:
    source_signal = np.random.randn(duration_samples)
    room_perturb.add_source(pos, signal=source_signal)
    
room_perturb.image_source_model(use_libroom=False)
room_perturb.compute_rir()


##TESTING STAGE
#Create room where test source positions are added
test_Set_xyz, test_Set_spherical = create_training(mic0_pos, m=M)


# Create the exact same room, where we wil add only the test positions 
room_test = pra.ShoeBox([6.5, 3.5, 2], fs=fs, max_order=3)
room_test.add_microphone_array(mic_array)

# Add wgn of 3 seconds duration 
duration_samples = 3*int(fs) 

for pos in test_Set_xyz:
    source_signal = np.random.randn(duration_samples)
    room_test.add_source(pos, signal=source_signal)
    
room_test.image_source_model(use_libroom=False)
room_test.compute_rir()