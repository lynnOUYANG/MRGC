# Multi-Relation Graph Clustering

# Available Data

You can download data from: <https://drive.google.com/file/d/1Pbeu_3WLIPUXlX5IJ36Vpz3vWCLSiSkK/view?usp=drive_link>

# Training

I put the best parameters in run.sh, you can reproduce the result by running `bash run.sh`

# Parameters Settings

|        | alpha | sigma (beta in object func) | n\_T (K) | dim  | seed | theta | max\_h | gamma (in ORF) |
| :----- | :---- | :-------------------------- | :------- | :--- | :--- | :---- | :----- | :------------- |
| acm    | 1     | 3                           | 6        | 128  | 6    | 160   | 2      | 0              |
| dblp   | 1     | 80                          | 4        | 64   | 42   | 2     | 1      | 0.1            |
| acm-2  | 1     | 5                           | 6        | 512  | 6    | 200   | 3      | 0.1            |
| yelp   | 1     | 34                          | 10       | 33   | 6    | 60    | 3      | 0              |
| amazon | 1     | 1                           | 4        | 65   | 42   | 15    | 3      | 0              |
| imdb   | 1     | 9                           | 18       | 1024 | 42   | 80    | 1      | 0.005          |
| mag    | 1     | 4                           | 10       | 64   | 42   | 100   | 40     | 0              |

# Envirorment Settings

python -- 3.9

torch -- 2.0.1+cu118

numpy -- 1.26.3

scipy -- 1.13.1

sklearn -- 1.5.2

torch-geometric -- 2.6.1

