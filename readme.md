#Recognize handwritten characters
The architecture of the MobileNetV2 neural network was used as a basis.
I used MobileNetV2 because it doesn't take up as much memory as ResNet.
Also, the task itself does not require such a deep neural network architecture, especially if we are talking about black and white images.
Of course, it would be possible to take a simpler model than MobileNetV2, such as LeNet-5.
But it seemed to me that it was better to take MobileNetV2 because it works with more classes.

Training was performed on the EMNIST_balance dataset.
Training accuracy on the test dataset was 98.43%. 

First, we set the appropriate image size for the neural network (224,224).

```python
 train_test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5], std=[0.5]),])
```
The first layer was modified to work with black and white images and the last layer because the dataset contains 47 classes.
```python
model = models.mobilenet_v2(weights=None) 
model.features[0][0]=nn.Conv2d(
    in_channels=1,    
    out_channels=32,
    kernel_size=3, 
    stride=2, 
    padding=1, 
    bias=False)
model.classifier[1] = nn.Linear(model.last_channel, 47)
nn.init.kaiming_normal_(model.classifier[1].weight, mode='fan_out', nonlinearity='relu')
```
Example of running the code:
`python inference.py --input mnt\test_data`

An example of the output:
```
python inference.py --input mnt\test_data
True
65/A------example_A.png
71/G------example_g.png
90/Z------example_Z.jpg
```


## Author
**Shtul Valentyn**  
- linkedin: https://www.linkedin.com/in/valentyn-s-7aab41309/
- Email: valentyn.yascovets@gmail.com
