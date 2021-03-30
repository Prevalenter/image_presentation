import torch
if __name__ == '__main__':
	# import urllib
	# from PIL import Image
	# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
	# try: urllib.URLopener().retrieve(url, filename)
	# except: urllib.request.urlretrieve(url, filename)
	# input_image = Image.open(filena
	x = torch.randn((1,3,256,256))
	print(x.shape, x.max())
	model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=False)
	model.eval()
	y = model(x)['out'][0]
	print(y.shape)

