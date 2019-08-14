
import numpy as np
from PIL import Image
import gym

from scipy import interpolate

from math import floor,sqrt

class CorrEnv(gym.Env):

    def __init__(self, ref_image, tar_image, poi, guess):
        self.subset_r = 5
        self.ref = ref_image
        self.tar = tar_image
        self.poi = poi
        
        self.p_list = []
        self.n_zncc_list = []
        
        self.p = guess
        self.p_list.append(self.p)
        self.n_zncc_list.append(1 - self.zncc())
        
        

    def step(self, action):
        
        self.p = self.p + action
        self.p_list.append(self.p)
        self.n_zncc_list.append(1 - self.zncc())
        
        done = False
        info = 'Nothing happened'
        return self.state, self.reward, done, info

    def reset(self):
        self.p_list = []
        self.n_zncc_list = []
        return self.state

    @property
    def state(self):
        return [self.p_list,self.n_zncc_list]

    @property    
    def reward(self):
        return -self.n_zncc_list[-1];



    def set_poi(self,poi):
        self.poi = poi

    def set_tar_image(self,tar_image):
        self.tar = tar_image

    def set_ref_image(self,ref_image):
        self.ref = ref_image


    def get_tar_pixel(self,r,c):
        return self.tar[r,c]

    def zncc(self):
        length = 2 * self.subset_r + 1
        row = self.poi[0] - self.subset_r
        col = self.poi[1] - self.subset_r

        ref_subset = self.ref[row:row+length, col:col+length]
        tar_subset = np.zeros(ref_subset.shape)
        
        p = self.p
        warp = np.array([[p[1]+1, p[2], p[0]],
                         [p[4], 1 + p[5], p[3]],
                         [0, 0, 1]])
                         
        # collect the points need interpolate
        # relative to poi, coors for target subset
        tar_coors = []
        x = y = self.subset_r
        for r in range(-y, y+1):
            for c in range(-x, x+1):
                tar_coor = np.matmul(warp, np.array([[c], [r], [1]]))
                cc = tar_coor[0,0]
                rr = tar_coor[1,0]
                tar_coor = np.array([rr,cc])
                tar_coor = tar_coor + self.poi
                tar_coors.append(tar_coor)
                # print(tar_coor, [c, r])
                
        tar_r = []
        tar_c = []        
        for coor in tar_coors:
            tar_r.append(coor[0])
            tar_c.append(coor[1])
        
        tar_r = np.array(tar_r)
        tar_c = np.array(tar_c)
        
        
        # calculate the range of data points for interploate 
        # a patch sampled from tar image 
        min_r = np.min(tar_r)
        max_r = np.max(tar_r)
        min_c = np.min(tar_c)
        max_c = np.max(tar_c)

        top = floor(min_r)
        left = floor(min_c)        
        h = floor(max_r - min_r + 5)
        w = floor(max_c - min_c + 5)
        lefttop = np.array([top,left])
        
        er,ec  = np.mgrid[0:h:1,0:w:1]
        er = er + top
        ec = ec + left
        tar_vals = self.get_tar_pixel(er,ec)
        
        
        
        
        # do interpolation
        func=interpolate.Rbf(er,ec,tar_vals,function='cubic')
        interp_vals = func(tar_r,tar_c)
        interp_vals = np.array(interp_vals).reshape(length,length) #careful reshape 
    
        # tar_subset got
        tar_subset = interp_vals
        
        
        # calculate zncc
        mean_ref = np.mean(ref_subset)
        mean_tar = np.mean(tar_subset)
        ref_de_mean = ref_subset -  mean_ref
        tar_de_mean = tar_subset -  mean_tar
        ref_ = sqrt(np.sum(ref_de_mean*ref_de_mean))
        tar_ = sqrt(np.sum(tar_de_mean*tar_de_mean))
        zncc = np.sum(ref_de_mean * tar_de_mean) / (ref_ * tar_)
        return zncc


def main():
    image_dir = "C:/Users/cslay/Downloads/20190715/images/"
    ref_file = "oht_cfrp_00.tiff"
    tar_file = "oht_cfrp_11.tiff"
    ref_image = Image.open(image_dir + ref_file)
    tar_image = Image.open(image_dir + tar_file)
    ref_array = np.array(ref_image)
    tar_array = np.array(tar_image)

    poi = np.array([880, 195])
    p0 = np.array([-0.401784,-0.000740886,0.00118266,-5.58845,-0.00082953,0.00484478])

    action = [0,0,0,0,0,0]

    env = CorrEnv(ref_array, tar_array, poi, p0)

    state = env.reset()
    for _ in range(3):
        state_, reward, done, info = env.step(action)
        action = np.random.rand(6)
        print('p_list',state_[0])
        print('n_zncc_list',state_[1])
        print('reward',reward)
        print()

    

if __name__ == '__main__':
    main()
