The files in this directory are used to create TFRecords based on MNIST datasets.

## MNIST dataset

create_tfrecords_str.py creates TFRecords with features encoded in numeric data types such as real numbers and integers.

create_tfrecords.py creates TFRecords with features encoded in numeric data types such as real and int.
create_tfrecords_str.py creates TFRecords with features encoded in string data types. This is to demonstrate how to create such a dataset because string encoding is better in terms of compression.



```python
# full resolution dataset
$ python create_tfrecords.py

# lowered resolution dataset
$ python create_tfrecords.py --resize 256
```
