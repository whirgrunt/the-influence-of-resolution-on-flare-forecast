# the-influence-of-resolution-on-flare-forecast
# Train and Test

```
$ CUDA_VISIBLE_DEVICES=0 python alexnet_scale.py --reduce_resolution 2 --cross True --cross_test_year 2010
```

--reduce_resolution represents the factor of reduction in resolution

--cross and --cross_test_year represents whether it is a cross-validation experiment, using the data of a certain year as a test, and the data of other years for training

#Visualization

$ python generate_cam_data.py 

#Draw ROC curve

$ python roc_draw.py 
