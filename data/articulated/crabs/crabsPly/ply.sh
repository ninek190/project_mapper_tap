for file1 in /home/euclid/ran/shapes/articulated/crabs/crabsPly/*.im;
do
file2=`echo $file1 | sed s/\.im$/\.ply/`
#set file2='echo $file1 | sed s/\.im/\.ply/'
echo $file2
/home/leonardo/ran/bin/obj2ply $file1 > $file2
done
