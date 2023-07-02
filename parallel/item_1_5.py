import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import time

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


item_1_5_par = SourceModule("""
#include <math.h>
__global__ void item_1_5_par(float *dest, float *m, float gamma, int linhas, int colunas){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int coord = ix+iy*linhas;
    if((ix < colunas) && (iy < linhas)){
		dest[coord] = pow((m[coord]/255),(1/gamma))*255;
    }
    if(dest[coord] < 0){
        dest[coord] = 0;
    }
    if(dest[coord] > 255){
        dest[coord] = 255;
    }
}
""")
                            
gamma_parallel = item_1_5_par.get_function("item_1_5_par")

in_folder = "../img_in/"
out_folder = '../img_out_parallel'

def imprime(img, modo='gray', vmin=0, vmax=255):
    plt.imshow(img, cmap=modo, vmin=vmin, vmax=vmax)
    plt.show()


###### item 1.5 Transformação de Brilho
def item_1_5():
    st = time.process_time()

    arquivo_1 = "Hi_Res.png"#input("Deseje o nome do arquivo desejado:\n")
    arquivo_1 = in_folder+arquivo_1

    img_1 = imageio.imread(arquivo_1)
    saida = np.zeros(np.shape(img_1))
    saida = saida.astype(np.float32)


    linhas = len(img_1)
    colunas = len(img_1[0])

    bdim = (32, 24, 1)
    dx, mx = divmod(linhas, bdim[0])
    dy, my = divmod(colunas, bdim[1])
    gdim = ( (dx + (mx>0)), (dy + (my>0)) )
    print(bdim, gdim)


    # temp = (img_1/255)
    plt.imshow(img_1, cmap='gray', vmin=0, vmax=255)
    plt.show()
    
    img_1 = img_1.astype(np.float32)
    saida = saida.astype(np.float32)

    gamma = 1.5
    gamma_parallel(drv.Out(saida), drv.In(img_1), np.float32(gamma), np.int32(linhas), np.int32(colunas), block=bdim, grid=gdim)
    plt.imshow(saida, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida = saida.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_5_out_gamma_1_5.png', saida)
    saida = saida.astype(np.float32)

    img_1 = img_1.astype(np.float32)
    saida = saida.astype(np.float32)

    gamma = 2.5
    gamma_parallel(drv.Out(saida), drv.In(img_1), np.float32(gamma), np.int32(linhas), np.int32(colunas), block=bdim, grid=gdim)
    plt.imshow(saida, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida = saida.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_5_out_gamma_2_5.png', saida)
    saida = saida.astype(np.float32)

    img_1 = img_1.astype(np.float32)
    saida = saida.astype(np.float32)

    gamma = 3.5
    gamma_parallel(drv.Out(saida), drv.In(img_1), np.float32(gamma), np.int32(linhas), np.int32(colunas), block=bdim, grid=gdim)
    plt.imshow(saida, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida = saida.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_5_out_gamma_3_5.png', saida)
    saida = saida.astype(np.float32)


    et = time.process_time()
    tt_time = et-st
    print("Elapsed", tt_time, "seconds on item_5")
    
item_1_5()

