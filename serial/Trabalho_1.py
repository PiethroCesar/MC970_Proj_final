import matplotlib.pyplot as plt
import numpy as np
import scipy
import imageio.v2 as imageio
import cv2
import time

in_folder = "../img_in/"
out_folder = '../img_out_serial'

def imprime(img, modo='gray', vmin=0, vmax=255):
    plt.imshow(img, cmap=modo, vmin=vmin, vmax=vmax)
    plt.show()


##### item 1.1
def item_1_1():
    st = time.process_time()

    arquivo_1 = "baboon.png"#input("Deseje o nome do arquivo desejado:\n")
    arquivo_1 = in_folder+arquivo_1

    img = imageio.imread(arquivo_1)

    # imprime(img)
    largura_1 = len(img)

    novo = [ img[x*int(largura_1/4):(x+1)*int(largura_1/4),y*int(largura_1/4):(y+1)*int(largura_1/4)]for x in range(4)for y in range(4)]


    aux = [5, 10, 12, 2, 7, 15, 0, 8, 11, 13, 1, 6, 3, 14, 9, 4] #String de montagem desejada pelo enunciado

    # aux = np.arange(0,16)
    # np.random.shuffle(aux) #usado para quando queremos a imagem aleatóriamente embaralhada

    linha = np.concatenate((np.concatenate((novo[aux[0]], novo[aux[1]], novo[aux[2]], novo[aux[3]]),axis=1), np.concatenate((novo[aux[4]], novo[aux[5]], novo[aux[6]], novo[aux[7]]),axis=1), np.concatenate((novo[aux[8]], novo[aux[9]], novo[aux[10]], novo[aux[11]]),axis=1), np.concatenate((novo[aux[12]], novo[aux[13]], novo[aux[14]], novo[aux[15]]),axis=1)), axis=0)

    # imprime(linha)
    linha = linha.astype(np.uint8)

    imageio.imsave(out_folder+'/item_1_1_out.png', linha)

    et = time.process_time()
    tt_time = et-st
    print("Elapsed", tt_time, "seconds on item_1")


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

    saida = img_1*0.2+img_2*0.8
    axarr[0].imshow(saida, cmap='gray', vmin=0, vmax=255)
    saida = saida.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_2_out_a.png', saida)

    saida = img_1*0.5+img_2*0.5
    axarr[1].imshow(saida, cmap='gray', vmin=0, vmax=255)
    saida = saida.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_2_out_b.png', saida)

    saida = img_1*0.8+img_2*0.2
    axarr[2].imshow(saida, cmap='gray', vmin=0, vmax=255)
    saida = saida.astype(np.uint8)

    plt.show()



    imageio.imsave(out_folder+'/item_1_2_out_c.png', saida)

    et = time.process_time()
    tt_time = et-st
    print("Elapsed", tt_time, "seconds on item_2")

###### item 1.3 Transformação de intensidade

def item_1_3():
    st = time.process_time()
    
    arquivo_1 = "city.png"#input("Deseje o nome do arquivo desejado:\n")
    arquivo_1 = in_folder+arquivo_1

    img_1 = imageio.imread(arquivo_1)


    saida_1 = img_1.astype(np.int16)
    saida_1 = np.subtract(saida_1, 255)
    saida_1 = np.absolute(saida_1)
    plt.imshow(saida_1, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida_1 = saida_1.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_3_out_a.png', saida_1)


    saida_2 = img_1.astype(np.float16)
    saida_2 = (img_1/255)*100+100
    saida_2 = saida_2.astype(np.int16)
    plt.imshow(saida_2, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida_2 = saida_2.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_3_out_b.png', saida_2)


    img_flip = np.flip(img_1, 1)

    saida_3 = img_1.astype(np.int16)

    for i in range(len(img_1)):
        if (i%2==0):
            saida_3[i] = img_flip[i]

    plt.imshow(saida_3, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida_3 = saida_3.astype(np.uint8)

    imageio.imsave(out_folder+'/item_1_3_out_c.png', saida_3)

    



    saida_4=img_1.astype(np.int16)

    largura_1 = len(img_1)
    metade_largura_1 = int(largura_1/2)
    for i in range(metade_largura_1):
        saida_4[metade_largura_1+i] = img_1[metade_largura_1-(i+1)]

    plt.imshow(saida_4, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida_4 = saida_4.astype(np.uint8)

    imageio.imsave(out_folder+'/item_1_3_out_d.png', saida_4)



    saida_5 = np.flip(img_1, 0)

    plt.imshow(saida_5, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida_5 = saida_5.astype(np.uint8)

    imageio.imsave(out_folder+'/item_1_3_out_e.png', saida_5)


    et = time.process_time()
    tt_time = et-st
    print("Elapsed", tt_time, "seconds on item_3")


###### item 1.4 Transformação de Cores

def item_1_4():
    st = time.process_time()

    arquivo_1 = "colorido.png"#input("Deseje o nome do arquivo desejado:\n")
    arquivo_1 = in_folder+arquivo_1

    img_1 = imageio.imread(arquivo_1)
    plt.imshow(img_1)
    plt.show()

    saida_final = np.empty(np.shape(img_1))

    saida_final[:,:,0] = img_1[:,:,0]*0.393 + img_1[:,:,1]*0.769 + img_1[:,:,2]*0.189
    saida_final[:,:,1] = img_1[:,:,0]*0.349 + img_1[:,:,1]*0.686 + img_1[:,:,2]*0.168
    saida_final[:,:,2] = img_1[:,:,0]*0.272 + img_1[:,:,1]*0.534 + img_1[:,:,2]*0.131
    saida_final[:,:,3] = img_1[:,:,3]

    saida_final[saida_final < 0 ] = 0
    saida_final[saida_final > 255] = 255
    saida_final = saida_final.astype('uint8')

    # saida_final[:,:,0] = [[int(np.sum(coluna[i]*[0.393, 0.769, 0.189, 0])) for i in range(len(coluna))]for coluna in img_1]
    # saida_final[:,:,1] = [[int(np.sum(coluna[i]*[0.349, 0.686, 0.168, 0])) for i in range(len(coluna))]for coluna in img_1]
    # saida_final[:,:,2] = [[int(np.sum(coluna[i]*[0.272, 0.534, 0.131, 0])) for i in range(len(coluna))]for coluna in img_1]
    # saida_final[:,:,3] = img_1[:,:,3]


    # for i in range(largura_1):
    #     for j in range(altura_1):
    #         saida[i][j] = [ np.sum(np.multiply(img_1[i][j], [0.393, 0.769, 0.189, 0])) , np.sum(np.multiply(img_1[i][j], [0.349, 0.686, 0.168, 0])), np.sum(np.multiply(img_1[i][j], [0.272, 0.534, 0.131, 0])), img_1[i][j][3]]
    plt.imshow(saida_final)
    plt.show()
    saida_final = saida_final.astype(np.uint8)

    imageio.imsave(out_folder+'/item_1_4_out_colorida.png', saida_final)


    saida_mono = np.empty((512, 512))
    saida_mono[:, :, ] = img_1[:,:,0]*0.2989+img_1[:,:,1]*0.5870+ img_1[:,:,2]*0.1140
    saida_mono[img_1[:,:,3] == 0] = (255)
    plt.imshow(saida_mono, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida_mono = saida_mono.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_4_out_monocromatica.png', saida_mono)

    et = time.process_time()
    tt_time = et-st
    print("Elapsed", tt_time, "seconds on item_4")


###### item 1.5 Transformação de Brilho


def item_1_5():
    st = time.process_time()

    arquivo_1 = "Hi_Res.png"#input("Deseje o nome do arquivo desejado:\n")
    arquivo_1 = in_folder+arquivo_1

    img_1 = imageio.imread(arquivo_1)

    gamma = 1.5

    temp = np.empty(np.shape(img_1))
    temp = (img_1/255)

    saida = np.empty(np.shape(img_1))
    saida = np.power(temp, (1/gamma))

    saida = saida*255
    saida[saida < 0 ] = 0
    saida[saida > 255] = 255

    plt.imshow(img_1, cmap='gray', vmin=0, vmax=255)
    plt.show()

    plt.imshow(saida, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida = saida.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_5_out_gamma_1_5.png', saida)


    gamma = 2.5
    saida = np.power(temp, (1/gamma))
    saida = saida*255
    saida[saida < 0 ] = 0
    saida[saida > 255] = 255
    plt.imshow(saida, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida = saida.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_5_out_gamma_2_5.png', saida)


    gamma = 3.5
    saida = np.power(temp, (1/gamma))
    saida = saida*255
    saida[saida < 0 ] = 0
    saida[saida > 255] = 255
    plt.imshow(saida, cmap='gray', vmin=0, vmax=255)
    plt.show()
    saida = saida.astype(np.uint8)
    imageio.imsave(out_folder+'/item_1_5_out_gamma_3_5.png', saida)

    et = time.process_time()
    tt_time = et-st
    print("Elapsed", tt_time, "seconds on item_5")


def item_1_6():
    st = time.process_time()

    arquivo_1 = "baboon.png"#input("Deseje o nome do arquivo desejado:\n")
    arquivo_1 = in_folder+arquivo_1

    img_1 = imageio.imread(arquivo_1)

    # imprime(img_1)

    saida_64 = img_1>>2
    saida_32 = saida_64>>1
    saida_16 = saida_32>>1
    saida_8 = saida_16>>1
    saida_4 = saida_8>>1
    saida_2 = saida_4>>1

    saida_64 = saida_64*4
    saida_32 = saida_32*8
    saida_16 = saida_16*16
    saida_8 = saida_8*32
    saida_4 = saida_4*64
    saida_2 = saida_2*128


    plt.figure(num='64 Níveis');
    # imprime(saida_64, vmin=None, vmax=None)
    plt.figure(num='32 Níveis');
    # imprime(saida_32, vmin=None, vmax=None)
    plt.figure(num='16 Níveis');
    # imprime(saida_16, vmin=None, vmax=None)
    plt.figure(num='8 Níveis'); 
    # imprime(saida_8, vmin=None, vmax=None)
    plt.figure(num='4 Níveis');
    # imprime(saida_4, vmin=None, vmax=None)
    plt.figure(num='2 Níveis');
    # imprime(saida_2, vmin=None, vmax=None)
    
    saida_64 = saida_64.astype(np.uint8)
    saida_32 = saida_32.astype(np.uint8)
    saida_16 = saida_16.astype(np.uint8)
    saida_8 = saida_8.astype(np.uint8)
    saida_4 = saida_4.astype(np.uint8)
    saida_2 = saida_2.astype(np.uint8)



    imageio.imsave(out_folder+'/item_1_6_out_64_niveis.png', saida_64)
    imageio.imsave(out_folder+'/item_1_6_out_32_niveis.png', saida_32)
    imageio.imsave(out_folder+'/item_1_6_out_16_niveis.png', saida_16)
    imageio.imsave(out_folder+'/item_1_6_out_8_niveis.png', saida_8)
    imageio.imsave(out_folder+'/item_1_6_out_4_niveis.png', saida_4)
    imageio.imsave(out_folder+'/item_1_6_out_2_niveis.png', saida_2)

    et = time.process_time()
    tt_time = et-st
    print("Elapsed", tt_time, "seconds on item_6")



def item_1_7_aux(img, i):
    plano_bit = np.empty(np.shape(img))
    plano_bit[:,:] = img[:,:]%2
    nome = 'Plano de bits '+str(i)
    plano_bit = plano_bit.astype(np.uint8)

    imageio.imsave(out_folder+'/item_1_7_out_'+nome+'.png', plano_bit)
    plt.figure(num=nome)
    # imprime(plano_bit, vmin=None, vmax=None)
    img = img>>1
    return img


def item_1_7():
    st = time.process_time()

    arquivo_1 = "baboon.png"#input("Deseje o nome do arquivo desejado:\n")
    arquivo_1 = in_folder+arquivo_1

    img_1 = imageio.imread(arquivo_1)

    for i in range(8):
        img_1 = item_1_7_aux(img_1, i)

    et = time.process_time()
    tt_time = et-st
    print("Elapsed", tt_time, "seconds on item_7")

def item_1_8():
    st = time.process_time()

    arquivo_1 = "baboon.png"#input("Deseje o nome do arquivo desejado:\n")
    arquivo_1 = in_folder+arquivo_1

    img_1 = imageio.imread(arquivo_1) 

    H1 = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])

    saida_1 = np.empty(np.shape(img_1))

    altura = np.shape(img_1)[0]
    largura = np.shape(img_1)[1]

    for x in range(altura):
        for y in range(largura):
            if(x>=2 and y>=2 and x<=altura-3 and y<=largura-3):
                saida_1[x][y] = np.sum(img_1[x-2:x+3,y-2:y+3]*H1)
            elif(x==1 and y>=2 and y<=largura-3):
                saida_1[x][y] = np.sum(img_1[x-1:x+3,y-2:y+3]*H1[1:])
            elif(x==altura-2 and y>=2 and y<=largura-3):
                saida_1[x][y] = np.sum(img_1[x-2:x+2,y-2:y+3]*H1[:-1])
            elif(x>=2 and y==1 and x<=altura-3):
                saida_1[x][y] = np.sum(img_1[x-2:x+3,y-1:y+3]*H1[:,1:])
            elif(x>=2 and y==largura-2 and x<=altura-3):
                saida_1[x][y] = np.sum(img_1[x-2:x+3,y-2:y+2]*H1[:,:-1])
            elif(x==1 and y==1):
                saida_1[x][y] = np.sum(img_1[x-1:x+3,y-1:y+3]*H1[1:,1:])
            elif(x==1 and y==largura-2):
                saida_1[x][y] = np.sum(img_1[x-1:x+3,y-2:y+2]*H1[1:,:-1])
            elif(x==altura-2 and y==1):
                saida_1[x][y] = np.sum(img_1[x-2:x+2,y-1:y+3]*H1[:-1,1:])
            elif(x==altura-2 and y==largura-2):
                saida_1[x][y] = np.sum(img_1[x-2:x+2,y-2:y+2]*H1[:-1,:-1])
            
            elif(x==0 and y>=2 and y<=largura-3):
                saida_1[x][y] = np.sum(img_1[x:x+3,y-2:y+3]*H1[2:])
            elif(x==altura-1 and y>=2 and y<=largura-3):
                saida_1[x][y] = np.sum(img_1[x-2:x+1,y-2:y+3]*H1[:-2])
            elif(x>=2 and y==0 and x<=altura-3):
                saida_1[x][y] = np.sum(img_1[x-2:x+3,y:y+3]*H1[:,2:])
            elif(x>=2 and y==largura-1 and x<=altura-3):
                saida_1[x][y] = np.sum(img_1[x-2:x+3,y-2:y+1]*H1[:,:-2])
            elif(x==0 and y==0):
                saida_1[x][y] = np.sum(img_1[x:x+3,y:y+3]*H1[2:,2:])
            elif(x==0 and y==largura-1):
                saida_1[x][y] = np.sum(img_1[x:x+3,y-2:y+1]*H1[2:,:-2])
            elif(x==altura-1 and y==0):
                saida_1[x][y] = np.sum(img_1[x-2:x+1,y:y+3]*H1[:-2,2:])
            elif(x==altura-1 and y==largura-1):
                saida_1[x][y] = np.sum(img_1[x-2:x+1,y-2:y+1]*H1[:-2,:-2])

            elif(x==0 and y == 1):
                saida_1[x][y] = np.sum(img_1[x:x+3,y-1:y+3]*H1[2:,1:])
            elif(x==0 and y == largura-2):
                saida_1[x][y] = np.sum(img_1[x:x+3,y-2:y+2]*H1[2:,:-1])
            elif(x==1 and y == 0):
                saida_1[x][y] = np.sum(img_1[x-1:x+3,y:y+3]*H1[1:,2:])
            elif(x==1 and y == largura-1):
                saida_1[x][y] = np.sum(img_1[x-1:x+3,y-2:y+1]*H1[1:,:-2])
            elif(x==altura-2 and y == 0):
                saida_1[x][y] = np.sum(img_1[x-2:x+2,y:y+3]*H1[:-1,2:])
            elif(x==altura-2 and y == largura-1):
                saida_1[x][y] = np.sum(img_1[x-2:x+2,y-2:y+1]*H1[:-1,:-2])
            elif(x==altura-1 and y == 1):
                saida_1[x][y] = np.sum(img_1[x-2:x+1,y-1:y+3]*H1[:-2,1:])
            elif(x==altura-1 and y == largura-2):
                saida_1[x][y] = np.sum(img_1[x-2:x+1,y-2:y+2]*H1[:-2,:-1])
            else:
                print('erro', x, y)

    saida_1[saida_1 > 255] = 255
    saida_1[saida_1 < 0] = 0

    # imprime(saida_1)
    saida_1 = saida_1.astype(np.uint8)
    imageio.imsave(out_folder+'/sitem_1_8_out_H1.png', saida_1)

    et = time.process_time()
    tt_time = et-st
    print("Elapsed", tt_time, "seconds on item_8")
    
# item_1_1()
# item_1_2()
# item_1_3()
# item_1_4()
item_1_5()
# item_1_6()
# item_1_7()
# item_1_8()



            

        



