from flask import Flask, jsonify, send_from_directory, make_response, abort
from flask import render_template
#from flask_cors import CORS
#from flask_cors import cross_origin
import os, datetime, re
from configparser import ConfigParser
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


cp = ConfigParser()
cp.read('./sql/mysql.conf')
host = cp.get("mysql", "db_host")
port = cp.getint("mysql", "db_port")
user = cp.get("mysql", "db_user")
password = cp.get("mysql", "db_password")
database = cp.get("mysql", "db_database")

VALID_INPUT = r'^chr\d{1,2}_\d{5,20}-\d{5,20}$'
ENGINE = "mysql+pymysql://{user}:{password}@{host}:{port}/{database}".format(user=user, password=password, host=host, port=port, database=database)

app = Flask(__name__, template_folder="./dist", static_folder="./dist/static")


@app.after_request
def delete_redundancy(res):
    dirToBeEmptied = ".\csv_file"
    ds = list(os.walk(dirToBeEmptied))
    delta = datetime.timedelta(seconds=300)
    now = datetime.datetime.now()
    for d in ds:
        os.chdir(d[0])
        if d[2] != []:
            for x in d[2]:
                ctime = datetime.datetime.fromtimestamp(os.path.getctime(x))
                if ctime < (now - delta):
                    os.remove(x)
    return res




@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', name='index')


@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html', name='home')


@app.route('/introduction', methods=['GET'])
def introduction():
    return render_template('introduction.html', name='introduction')


@app.route('/results', methods=['GET', 'POST'])
def results():
    # gene = request.args.get('gene')
    return render_template('results.html', name='results')


@app.route('/results/<file_name>', methods=['GET'])
#@cross_origin()
def query_results_table(file_name):
    try:
        file_name = file_name.replace(".csv", "")
        if (re.match(VALID_INPUT, file_name, flags=0) == None):
            app.logger.info('invalid request')
            abort(404)
        loc_chr = int(file_name.split('_')[0].split('chr')[1])
        start_loc = int(file_name.split('_')[1].split('-')[0])
        end_loc = int(file_name.split('_')[1].split('-')[1])
    except Exception as err:
        print(err)
        app.logger.error(err)
        abort(404)
    try:
        if(end_loc-start_loc<=300):
            engine = create_engine(ENGINE)
            sql = "select target_sequence, genomic_location, strand, GC_content, self_complementarity, MM0, MM1, MM2, MM3, efficiency from res where (location_num>={0} and location_num<={1} and location_chr={2}) order by efficiency desc;" .format(start_loc, end_loc, loc_chr)
            df = pd.read_sql_query(sql, engine)
            col_name = df.columns.tolist()
            col_name.insert(0, 'Rank')
            df = df.reindex(columns=col_name)
            df['Rank'] = df.index + 1
            df.columns = ['Rank', 'Target sequence', 'Genomic location', 'Strand', 'GC content (%)', 'Self-complementarity',
                          'MM0', 'MM1', 'MM2', 'MM3', 'Efficiency']
            df = df.reset_index(drop=True)
            directory = os.getcwd()
            df.to_csv(directory + '/' + file_name + '.csv', index=None)
            response_tsv = make_response(send_from_directory(directory, file_name + '.csv', as_attachment=True))
        else:
            abort(404)


    except Exception as err:
        print(err)
        app.logger.error(err)
        abort(404)

    return response_tsv


@app.route('/picture/<file_name>', methods=['POST'])
#@cross_origin()
def query_picture_contents(file_name):
    try:
        if (re.match(VALID_INPUT, file_name, flags=0) == None):
            app.logger.info('invalid request')
            abort(404)
        loc_chr = int(file_name.split('_')[0].split('chr')[1])
        start_loc = int(file_name.split('_')[1].split('-')[0])
        end_loc = int(file_name.split('_')[1].split('-')[1])
        if (end_loc - start_loc <= 300):
            engine = create_engine(ENGINE)
            sql = "select location_num, efficiency, strand, TSS_loc from res where (location_num>={0} and location_num<={1} and location_chr={2}) order by efficiency desc;".format(
                start_loc, end_loc, loc_chr)
            df = pd.read_sql_query(sql, engine)

            col_name = df.columns.tolist()
            col_name.insert(0, 'Rank')

            df = df.reindex(columns=col_name)
            df['Rank'] = df.index + 1

            if len(df)>0:
                loc_array = df['location_num'].values
                loc_array.sort()
                layer_dict = {}
                layer_dict.update({loc_array[0]: 0})
                layer = 0
                current_loc = int(loc_array[0]) + 23
                loc_array = np.delete(loc_array, 0)
                while (int(loc_array.shape[0]) > 0):
                    del_list = []
                    for i in range(loc_array.shape[0]):
                        if (int(loc_array[i]) > current_loc):
                            del_list.append(i)
                            current_loc = int(loc_array[i]) + 23
                            layer_dict.update({loc_array[i]: layer})
                    layer += 1
                    loc_array = np.delete(loc_array, del_list)
                    if (loc_array.shape[0] > 0):
                        layer_dict.update({loc_array[0]: layer})
                        current_loc = int(loc_array[0]) + 23
                        loc_array = np.delete(loc_array, 0)
                cutcoords = []
                for i in range(len(df)):
                    cutcoords.append([int(df['Rank'][i]), int(df['location_num'][i]), df['efficiency'][i], 23, df['strand'][i],
                                      int(layer_dict[df['location_num'][i]])])
                tss = [int(df['TSS_loc'][0])]
                genemin = int(df['location_num'].min(axis=0))
                genemax = int(df['location_num'].max(axis=0) + 23)
                batch_coor = [{"chromo": file_name.split('_')[0], "zone":[[file_name.split('_')[0], start_loc, end_loc]]}]
                data = {'cutcoords': cutcoords, 'tss': tss, 'genemin': genemin, 'genemax': genemax, 'batch_coor': batch_coor, 'msg': '666'}
            else:
                data = {'cutcoords': '', 'tss': '', 'genemin': '', 'genemax': '', 'batch_coor': '', 'msg': '687'}
        else:
            data = {'cutcoords': '', 'tss': '', 'genemin': '', 'genemax': '', 'batch_coor': '', 'msg': '687'}

    except Exception as err:
        app.logger.error(err)
        abort(404)

    return make_response(jsonify(data), 200)


@app.route('/test/<file_name>', methods=['POST'])
#@cross_origin()
def test_if_exits(file_name):
    try:
        if (re.match(VALID_INPUT, file_name, flags=0) == None):
            app.logger.info('invalid request')
            abort(404)
        loc_chr = int(file_name.split('_')[0].split('chr')[1])
        start_loc = int(file_name.split('_')[1].split('-')[0])
        end_loc = int(file_name.split('_')[1].split('-')[1])

        engine = create_engine(ENGINE)
        sql = "select efficiency from res where (location_num>={0} and location_num<={1} and location_chr={2}) order by efficiency desc;".format(
            start_loc, end_loc, loc_chr)
        df = pd.read_sql_query(sql, engine)

        if len(df)>0:
            data = {'msg': '666'}
        else:
            data = {'msg': '687'}
    except Exception as err:
        print(err)
        app.logger.error(err)
        abort(404)
    print(data)
    return make_response(jsonify(data), 200)



if __name__ == '__main__':
    #CORS(app)
    app.debug = True

    handler = RotatingFileHandler('log/logs.log', maxBytes=1024*1024, backupCount=10, encoding='UTF-8')
    handler.setLevel(logging.DEBUG)
    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
    handler.setFormatter(logging_format)
    app.logger.addHandler(handler)
    app.run(host="0.0.0.0", port="3000", debug=True)
