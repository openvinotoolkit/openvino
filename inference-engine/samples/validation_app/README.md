# Validation App {#InferenceEngineValidationApp}

Inference Engine Validation Application ("validation app" for short) is a tool that allows the user to score common topologies with 
de facto standard inputs and outputs configuration. Such as AlexNet or SSD. Validation app allows the user to collect simple 
validation metrics for the topologies. It supports Top1/Top5 counting for classification networks and 11-points mAP calculation for
object detection networks.

Possible usages of the tool:
* Check if Inference Engine scores the public topologies well (the development team uses the validation app for regular testing and the user bugreports are always welcome)
* Verify if the user's custom topology compatible with the default input/output configuration and compare its accuracy with the public ones
* Using Validation App as another sample: although the code is much more complex than in classification and object detection samples, it's still open and could be re-used

This document describes the usage and features of Inference Engine Validation Application.

## Validation Application options

Let's list <code>validation_app</code> CLI options and describe them:

	Usage: validation_app [OPTION]
	
	Available options:
	
	    -h                        Print a usage message
	    -t <type>                 Type of the network being scored ("C" by default)
	      -t "C" for classification
	      -t "OD" for object detection
	    -i <path>                 Required. Folder with validation images, folders grouped by labels or a .txt file list for classification networks or a VOC-formatted dataset for object detection networks
	    -m <path>                 Required. Path to an .xml file with a trained model
	    -l <absolute_path>        Required for MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernel implementations
	    -c <absolute_path>        Required for clDNN (GPU)-targeted custom kernels.Absolute path to the xml file with the kernel descriptions
	    -d <device>               Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)
	    -b N                      Batch size value. If not specified, the batch size value is determined from IR
	    -ppType <type>            Preprocessing type. One of "None", "Resize", "ResizeCrop"
	    -ppSize N                 Preprocessing size (used with ppType="ResizeCrop")
	    -ppWidth W                Preprocessing width (overrides -ppSize, used with ppType="ResizeCrop")
	    -ppHeight H               Preprocessing height (overrides -ppSize, used with ppType="ResizeCrop")
	    --dump                    Dump filenames and inference results to a csv file
	
	    Classification-specific options:
	      -Czb true               "Zero is a background" flag. Some networks are trained with a modified dataset where the class IDs are enumerated from 1, but 0 is an undefined "background" class (which is never detected)
	
	    Object detection-specific options:
	      -ODkind <kind>          Kind of an object detection network: SSD
	      -ODa <path>             Required for OD networks. Path to the folder containing .xml annotations for images
	      -ODc <file>             Required for OD networks. Path to the file containing classes list
	      -ODsubdir <name>        Folder between the image path (-i) and image name, specified in the .xml. Use JPEGImages for VOC2007

 
There are three categories of options here.
1. Common options, usually named with a single letter or word, such as <code>-b</code> or <code>--dump</code>. These options have a common sense in all validation_app modes.
2. Network type-specific options. They are named as an acronym of the network type (such as <code>C</code> or <code>OD</code>, followed by a letter or a word addendum. These options are specific for the network type. For instance, <code>ODa</code> option makes sense only for an object detection network.

Let's show how to use Validation Application in all its common modes

## Running classification

This topic demonstrates how to run the Validation Application, in classification mode to score a classification CNN on a pack of images.

You can use the following command to do inference of a chosen pack of images:
```bash
./validation_app -t C -i <path to images main folder or .txt file> -m <model to use for classification> -d <CPU|GPU>
```

### Source dataset format: folders as classes
A correct bunch of files should look something like:

	<path>/dataset
		/apron
			/apron1.bmp
			/apron2.bmp
		/collie
			/a_big_dog.jpg
		/coral reef
			/reef.bmp
		/Siamese
			/cat3.jpg

To score this dataset you should put `-i <path>/dataset` option to the command line

### Source dataset format: a list of images
Here we use a single list file in the format "image_name-tabulation-class_index". The correct bunch of files:

	<path>/dataset
		/apron1.bmp
		/apron2.bmp
		/a_big_dog.jpg
		/reef.bmp
		/cat3.jpg
		/labels.txt
		
where `labels.txt` looks like:

	apron1.bmp 411
	apron2.bmp 411
	cat3.jpg 284
	reef.bmp 973
	a_big_dog.jpg 231

To score this dataset you should put `-i <path>/dataset/labels.txt` option to the command line

### Outputs

Progress bar will be shown representing a progress of inference.
After inference is complete common info will be shown:
<pre class="brush: bash">
Network load time: time spent on topology load in ms
Model: path to chosen model
Model Precision: precision of the chosen model
Batch size: specified batch size
Validation dataset: path to a validation set
Validation approach: Classification networks
Device: device type
</pre>
Then application shows statistics like average infer time, top 1 and top 5 accuracy, for example:
<pre class="brush: bash">
Average infer time (ms): 588.977 (16.98 images per second with batch size = 10)

Top1 accuracy: 70.00% (7 of 10 images were detected correctly, top class is correct)
Top5 accuracy: 80.00% (8 of 10 images were detected correctly, top five classes contain required class)
</pre>

### How it works

Upon the start-up the validation application reads command line parameters and loads a network to the Inference Engine plugin.
Then program reads validation set (<code>-i</code> option): 

- If it specifies a directory, the program tries to load labels first. To do this, app searches for the file with the same name as model but with ".labels" extension (instead of .xml).
  Then it searches the folder specified and adds to the validation set all images from subfolders whose names are equal to some known label. 
  If there are no subfolders whose names are equal to known labels, validation set is considered to be empty.
  
- If it specifies a .txt file, the application reads file considering every line has an expected format: <code>&lt;relative_path_from_txt_to_img&gt; &lt;ID&gt;</code>.
  <code>ID</code> is a number of image that network should classify.

After that, the app reads the number of images specified by the <code>-b</code> option and loads them to plugin.

<strong>Note:</strong> Images loading time is not a part of inference time reported by the application.

When all images are loaded, a plugin executes inferences and the Validation Application collects the statistics.

It is possible to retrieve infer result by specifying <code>--dump</code> option. 

This option enables creation (if possible) of an inference report with the name in format <code>dumpfileXXXX.csv</code>.

The structure of the report is a number of lines, each contains semicolon separated values:
* image_path;
* flag representing correctness of prediction;
* id of Top-1 class;
* probability that the image belongs to Top-1 class;
* id of Top-2 class;
* probability that the image belongs to Top-2 class;
* etc.

## Object detection

This topic demonstrates how to run the Validation Application, in object detection mode to score an object detection 
CNN on a pack of images.

Validation app was validated with SSD CNN. Any network that can be scored by the IE and has the same input and output 
format as one of these should be supported as well.

### Running SSD on the VOC dataset

SSD could be scored on the original dataset that was used to test it during its training. To do that:

1. From the SSD author's page (<code>https://github.com/weiliu89/caffe/tree/ssd</code>) download Pre-trained SSD-300: 
	
	https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA

2. Download VOC2007 testing dataset (this link could be found on the same github page): 
  ```bash
  $wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  tar -xvf VOCtest_06-Nov-2007.tar
  ```
3. Convert the model with Model Optimizer

4. Create a proper class file (made from the original <code>labelmap_voc.prototxt</code>)

	none_of_the_above 0
	aeroplane 1
	bicycle 2
	bird 3
	boat 4
	bottle 5
	bus 6
	car 7
	cat 8
	chair 9
	cow 10
	diningtable 11
	dog 12
	horse 13
	motorbike 14
	person 15
	pottedplant 16
	sheep 17
	sofa 18
	train 19
	tvmonitor 20

...and save it as <code>VOC_SSD_Classes.txt</code> file.

5. Score the model on the dataset:
  ```bash
  ./validation_app -d CPU -t OD -ODa "<...>/VOCdevkit/VOC2007/Annotations" -i "<...>/VOCdevkit" -m "<...>/vgg_voc0712_ssd_300x300.xml" -ODc "<...>/VOC_SSD_Classes.txt" -ODsubdir JPEGImages
  ```
As a result you should see a progressbar that will count from 0% to 100% during some time and then you'll see this:

	Progress: [....................] 100.00% done    
	[ INFO ] Processing output blobs
	Network load time: 27.70ms
	Model: /home/user/models/ssd/withmean/vgg_voc0712_ssd_300x300/vgg_voc0712_ssd_300x300.xml
	Model Precision: FP32
	Batch size: 1
	Validation dataset: /home/user/Data/SSD-data/testonly/VOCdevkit
	Validation approach: Object detection network
	
	Average infer time (ms): 166.49 (6.01 images per second with batch size = 1)
	Average precision per class table: 
	
	Class	AP
	1	0.796
	2	0.839
	3	0.759
	4	0.695
	5	0.508
	6	0.867
	7	0.861
	8	0.886
	9	0.602
	10	0.822
	11	0.768
	12	0.861
	13	0.874
	14	0.842
	15	0.797
	16	0.526
	17	0.792
	18	0.795
	19	0.873
	20	0.773
	
	Mean Average Precision (mAP): 0.7767

This value (Mean Average Precision) is specified in a table on the SSD author's page (<code>https://github.com/weiliu89/caffe/tree/ssd</code>) and in their arXiv paper (http://arxiv.org/abs/1512.02325)

## See Also
 
* [Using Inference Engine Samples](@ref SamplesOverview)
