# ML Final Project

Tensorflow 2

## Part1

### Changes Made:
1. Changed Dropouts to 0.5 due to generally giving better performance.

2. Changed epochs to 200 from 100. Allows it to train longer and better.

3. Changed optimizer to Adamax, which is an imporovement to Adam, which is an improvement on RMSprop.

4. Added batch normalization after convolution layers.

5. Made decay rate smaller to keep learning rate change lower.

6. No data augmentation, kept lowering my performance.

**Results:** 83% to 84%

### Run on GAIVI
```bash
# qsub script.py GPU#
qsub part1.sh 6

# Check output
cat output_part1.txt
```

## Part2

### Changes Made:
1. Took original model and made fully convolutional without a single dense layer.

2. Gets same performance as original but now I can run much larger images through it like in the horse test which is not 32x32x3 without resizing.

3. Created my own FCN ResNet to get best perforamnce.

**Results Original FCN:** 83% to 84%, same as original non-FCN

**Results ResNet FCN:** Didnt get results in time

### Run on GAIVI
```bash
# qsub script.py GPU#
qsub part2.sh 6

# Check output 
cat output_part2.txt
```