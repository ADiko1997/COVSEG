
import matplotlib as plt 
import visdom

#@title plotting for training
class LinePlotter(object):
    def __init__(self, env_name="main"):
        self.vis = visdom.Visdom(use_incoming_socket=False)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x, x]),
                                    Y=np.array([y, y]), env=self.env, opts=dict(
                                    legend=[split_name],
                                    title=var_name,
                                    xlabel="Iters",
                                    ylabel=var_name
                                    ))
        else:
            self.vis.updateTrace(X=np.array([x, x]), Y=np.array([y, y]), env=self.env,
                                win=self.plots[var_name], name=split_name)



def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
    print(img.shape)
    img = img.permute(1, 2, 0)
    plt.figure(figsize = (20, 20))
    plt.imshow(img)
    plt.show()
 
def imshow_mask(img):
    img = img * 50     # unnormalize
    img = img.permute(1, 2, 0)
    plt.figure(figsize = (20, 20))
    plt.imshow(img)
    plt.show()
    
"""
#Syntax how to use matplotlib visualizer
dataiter = iter(val_loader)
images, labels_, im, masks_ = dataiter.next()
images_p = images[:4]
labels_p = labels_[:4]
images = images.permute(0,3,1,2)
labels_ = labels_.view(10,512,512,1)
labels_ = labels_.permute(0,3,1,2)
print(labels_p.shape)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
imshow_mask(torchvision.utils.make_grid(labels_))
"""


