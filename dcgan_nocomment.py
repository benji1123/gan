#deep conv Gans
###################
# C410 (famous dataset)  images 
; 400 px in from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

if __name__ == "__main__":
    #setting hyperparameters (pre-training)
    batchSize = 64  #set batch-size to 64 TEs per iter.
    imageSize = 64  #clarify image-size as 64x64 px
    #setting transformation-specs for TE's
    transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.    
    # Loading the dataset
    dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch
   
    # Takes NN as input | initialize all its weights...
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)        # set weights for Conv-layers
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)        # set weights for BatchNorm-layers
            m.bias.data.fill_(0)
            
    #........................ Defining "G" - Generator........................
    class G(nn.Module): 
        def __init__(self): # We introduce the __init__() function that will define the architecture of the generator.........
            
            super(G, self).__init__() # We inherit from the nn.Module tools. 
            # creating DCNN Architecture: sequence of modules (convolutions, ReLu, etc.)......
            self.main = nn.Sequential( 
            # using an industry-known architecture
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),   # inversed conv. | (input-sz, output-sz (# ft.maps), kernal-size, stride, padding)
                nn.BatchNorm2d(512),                                   # normalize all the features along the dimension of the batch
                nn.ReLU(True),                                         # apply a ReLU rectification to break the linearity
                # applying 4 more sets of Conv/Batch/ReLu...
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), 
                nn.BatchNorm2d(256), 
                nn.ReLU(True), 
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128), 
                nn.ReLU(True), 
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(True), 
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False), # We add another inversed convolution.
                nn.Tanh() # We apply a Tanh rectification to break the linearity and stay between -1 and +1.
            )
        # Implementing forward-prop | return output containing the generated images......
        def forward(self, input): 
            output = self.main(input) # We forward propagate the signal through the whole neural network of the generator defined by self.main.
            return output # We return the output containing the generated images.
    
    #......... Creating the generator......
    netG = G()                  
    netG.apply(weights_init)    # setting weights 
    
    #........................ Defining "D" - Discriminator........................
    class D(nn.Module): 
        def __init__(self):
            super(D, self).__init__()
            # Creating CNN Architecture......
            self.main = nn.Sequential(
                    nn.Conv2d(3,64,4,2,1,bias=False),    #apply convolutions in reverse comp. to "G"
                    nn.LeakyReLU(0.2,inplace=True),      #leakyReLu works better than simple ReLu here  
                    nn.Conv2d(64, 128, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(128, 256, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(256,512,4,2,1,bias=False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True), 
                    nn.Conv2d(512, 1, 4, 1, 0, bias = False),                     
                    nn.Sigmoid() #Sigmoid rectification to break linearity and limit output (score) to btwn 0 & 1
            )
            # Takes Generated Img (input) | Return "D" Score
            def forward(self,input):
                output = self.main(input)
                return output.view(-1)     # .view(-1) flattens CNN output

    #......... Creating the discriminator.........
    netD = D()                  
    netD.apply(weights_init)    # setting weights 
    
    #..................Training DCGANs (Deep Convolutional GANs)..................
    criterion = nn.BCELoss()    #BCE (binary cross entropy) loss between target & output
    #parameter (theta []) opimizers 
    optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas=[0.5,0.999])      #disciminator
    optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas=[0.5,0.999])      #generator
    
        
    # .........Implementing Backpropogation........
    for epoch in range(25):         #accessing ground-truth-images from dataset | Discriminator
        for i, data in enumerate(dataloader, 0):        #"dataloader" object is defined Ln 29 | contains data in batches
            
            #1 : update weights of D (= zero in relation to Grad)
            netD.zero_grad()                        
            #1.2 : Sending Real-img into D | Getting Error 
            real, _ = data                                      # tensor object 
            input = Variable(real)                              # put Input in (required) torch-Variable form for NN 
            target = Variable(torch.ones(input.size()[0]))      
            output = netD(input)                                # input --> Discriminator NN --> output
            errD_real = criterion(output,target)                # get BCE error of the "realness" score given to Real-img
            
            #1.3 : Getting Error of Gen image
            noise = Variable(torch.randN(input.size()[0],100,1,1))         # generator input...
            #[batch-size, input-size, ftmap-w, ftmap-h]
            fake = netG(noise)                                             # fake-images (generator output)
            target = Variable(torch.zeros(input.size()[0]))                 
            output = netD(fake.detatch())                                  # "realness" score of fake-imgs from D
            errD_fake = criterion(output,target)
                   
            #1.4 : backpropogate total error  
            errD = errD_fake + errD_real
            errD.backward()                     #backpropogate error
            optimizerD.step()                   #optimizerD is predefined near top
            
            # 2 : update weights of G | Getting Error    
            netG.zero_grad()                                    # no idea why
            target = Variable(torch.ones(input.size()[0]))
            output = netD(fake)                                 # purported to update weights | do not detatch grad
            errG = criterion(output,target)
            
            #backpropogate
            errG.backward()
            optimizerG.step()                  #optimizerG is predefined near top
            
            # 3 : printing run-time data
            print("[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f" % (epoch,25,i,len(dataloader),errD.data[0],errG.data[0]))
            #save every 100th minibatch
            if i % 100 == 0:
                vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True) 
                fake = netG(noise) # We get our fake generated images.
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True) 