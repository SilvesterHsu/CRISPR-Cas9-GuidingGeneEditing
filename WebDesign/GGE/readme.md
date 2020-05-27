# GGE

GGE is a web application for predicting the gene knock-out efficiency using Crispr/Cas 9 on zebra fish. Predicting is based on transfer learning and for details of the algorithm, please refer to the paper.

## Deployment Instructions

### GGE File Directory

GGE

- .idea
- csv_file
- dist
- log
  - logs.log
- module_installation
  - pymodule_ini.py
- sql
  - GGE_data_backup
    - gge_res
    - gge_rountines
  - mysql.conf
- app.py
- readme.md

### Operating Environment

Please install MySQL 5.7 and python 3.x.

### Operation Steps

1. Install MySQL 5.7 and python 3.X.
2. Execute gge_res.sql in ./sql/GGE_data_backup to create the data base and import the prediction results.
3. Run pymodule_ini.py to install dependent python modules that will be used in this project.
4. Change the configurations in ./sql/mysql.conf to your own.
5. Run app.py, the default port is 3000, if you want any change, please modify `app.run(host="0.0.0.0", port="3000", debug=True)` in app.py.

## APIs

| URL                  | method    | return                                                       | function                                |
| -------------------- | --------- | ------------------------------------------------------------ | --------------------------------------- |
| /home                | GET, POST | html                                                         | query page                              |
| /introduction        | GET       | html                                                         | introduction page                       |
| /results             | GET, POST | html                                                         | result page                             |
| /results/<file_name> | GET       | csv file                                                     | to download the result in csv format    |
| /picture/<file_name> | POST      | {'cutcoords': '', 'tss': '', 'genemin': '', 'genemax': '', 'batch_coor': '', 'msg': '687'} (json format) | to get data for visualization           |
| /test/<file_name>    | POST      | {'msg': '666'/'687'} (json format)                           | to verify whether the query has results |

## Other

The web front end is written via Vue.js framework and is compiled and minified into dist folder in this project. To see its source, please refer to GGE_front.zip. If you want to modify the code of the web front end, please install node.js first then unzip GGE_front.zip to make any changes. Once finishing the modification, you need to enter that directory and run `npm run build`, then copy the dist folder to replace the dist folder in GGE.

