data=3rdparty/housegan/FloorplanDataset/eval_data.npy
args="-m civl5220_group13.__init__ housegan constraint_inference $data"

# python $args --contour ./evaluation/contours/irregular.txt --output irregular
# python $args --contour ./evaluation/contours/rectangle.txt --output rectangle

python $args --contour ./evaluation/contours/irregular.txt --output irregular-gcs --criterions nlccs ciou
python $args --contour ./evaluation/contours/rectangle.txt --output rectangle-gcs --criterions nlccs ciou

python $args --contour ./evaluation/contours/irregular.txt --output irregular-gqs --criterions nlccs fiou
python $args --contour ./evaluation/contours/rectangle.txt --output rectangle-gqs --criterions nlccs fiou
