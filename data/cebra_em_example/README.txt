
This is the workflow that I ran with the test data:


```
# 1. Initialize the project (just use default parameters)
init_project.py ./proj_10nm /path/to/20140801_hela-wt_xy5z8nm_as_crop_xy512z256px.xml
#    When done, you can load the project into Mobie using Fiji -> Plugins>MoBIE>Open MoBIE Project ... 
#    It asks for a location where you put "/path/to/the/proj_10nm/data"
#    Whenever anything is updated by the scripts below, you can refresh the respective map by removing it from view ('X') and adding it again ('view')

# 2. Navigate to the project
cd proj_10nm

# 3. Run membrane prediction and supervoxels
run.py -t supervoxels

# 4. Initialize a ground truth cube at the center of the dataset which is only 128 pixels in each dimension (usually one would use the default 256 px)
init_gt.py -b "1.28 1.28 1.024" -s 128 128 128

# 5. Extract the data for ground truth annotation
run.py -t gt_cubes

# 6.a) Now annotate the data for mitochondria using CebraANN
napari -w cebra-ann
#    Click on "Load Project", navigate to gt/gt000 and click "Open"
#    Annotated for the mitochondria and don't forget to put them into a semantic layer, then click "Save"
#    Close Napari and continue in the terminal

# 6.b) Alternatively to (a) you can copy the mito.h5 from example_proj/gt/gt000 to the folder gt/gt000 in your project and proceed to step 7. 
#    However, note that this mito annotation is specifically for the position from step 4. 

# 7. Initialize a segmentation map
init_segmentation.py mito it00
#    Change the batch_shape to [128, 128, 128] when prompted for the parameters. 
#    Normally I would keep the defaults but we want to run a stitching later on and otherwise there is nothing to stitch...

# 8. Link the ground truth to the segmentation
link_gt.py 0 mito mito_it00

# 9. To check if this worked, run
log_gt.py -d 
#    Which should show something like this:

OUTPUT >>>>
____________________________________________________________________________________
DATASET = mito_it00

  TRAIN         ID      Layer   Annotated

      gt000     0       mito    yes

  VAL           ID      Layer   Annotated

      Not linked to any ground truth cubes.

____________________________________________________________________________________
<<<<

# 10. Run the segmentation
run.py -t mito_it00
#     In order to show mito_it00 in MoBIE, click 'view' in the dataset row, then it can be found under 'segmentations'
#     Same applies to the stitched map from step 11 below

# 11. Run the stitching
run.py -t stitch-mito_it00 --param beta=0.5


```


