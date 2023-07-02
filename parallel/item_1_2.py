import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import time

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

item_1_2_par = SourceModule("""
__global__ void item_1_2_par(float *dest, float *m, float *n, float a, float b, int linhas, int colunas){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int coord = ix+iy*colunas;
    if((ix <= colunas) && (iy <= linhas)){
        dest[coord] = m[coord]*a+n[coord]*b;
    }
}
""")
                            
prop_parallel = item_1_2_par.get_function("item_1_2_par")

in_folder = "../img_in/"
out_folder = '../img_out_parallel'

def imprime(img, modo='gray', vmin=0, vmax=255):
    plt.imshow(img, cmap=modo, vmin=vmin, vmax=vmax)
    plt.show()

###### item 1.2 Combinação de imagens
def item_1_2():
    st = time.process_time()
    

    arquivo_1 = "baboon.png"#input("Deseje o nome do arquivo desejado:\n")
    arquivo_1 = in_folder+arquivo_1
    img_1 = imageio.imread(arquivo_1)

    arquivo_2 = "butterfly.png"#input("Deseje o nome do arquivo desejado:\n")
    arquivo_2 = in_folder+arquivo_2
    img_2 = imageio.imread(arquivo_2)

    f, axarr = plt.subplots(1,3) 

    saida = np.ones(img_1.shape)

    linhas = len(img_1)
    colunas = len(img_1[0])
    

    bdim = (32, 24, 1)
    dx, mx = divmod(linhas, bdim[0])
    dy, my = divmod(colunas, bdim[1])
    gdim = ( (dx + (mx>0)), (dy + (my>0)) )
    
    print(bdim, gdim)
    img_1 = img_1.astype(np.float32)
    img_2 = img_2.astype(np.float32)
    saida = saida.astype(np.float32)


    prop_parallel(drv.Out(saida), drv.In(img_1), drv.In(img_2), np.float32(0.2), np.float32(0.8), np.int32(linhas), np.int32(colunas), block=bdim, grid=gdim)
    axarr[0].imshow(saida, cmap='gray', vmin=0, vmax=255)
    saida = saida.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_2_out_a.png', saida)
    saida = saida.astype(np.float32)
    
    prop_parallel(drv.Out(saida), drv.In(img_1), drv.In(img_2), np.float32(0.5), np.float32(0.5), np.int32(linhas), np.int32(colunas), block=bdim, grid=gdim)
    axarr[1].imshow(saida, cmap='gray', vmin=0, vmax=255)
    saida = saida.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_2_out_b.png', saida)
    saida = saida.astype(np.float32)

    prop_parallel(drv.Out(saida), drv.In(img_1), drv.In(img_2), np.float32(0.8), np.float32(0.2), np.int32(linhas), np.int32(colunas), block=bdim, grid=gdim)
    axarr[2].imshow(saida, cmap='gray', vmin=0, vmax=255)
    saida = saida.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_2_out_c.png', saida)

    plt.show()

    et = time.process_time()
    tt_time = et-st
    print("Elapsed", tt_time, "seconds on item_2")

item_1_2()
