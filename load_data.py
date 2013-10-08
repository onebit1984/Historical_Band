"""
Author: Vaibhav Bhandari <vb2317@columbia.edu>

"""

import pyodbc
import datetime as dt

def read_db(driver='{SQL Server Native Client 10.0}',server='bell.ldeo.columbia.edu',db='RUDIN_345PARK',table='RUDINSERVER_CURRENTSTEAMDEMAND',uid='vb2317',pwd='BELLvb@#17'):
    driver = '/usr/lib/i386-linux-gnu/odbc/libtdsodbc.so'
    #conn_str = "DRIVER=%s;SERVER=%s;DATABASE=%s;UID=%s;PWD=%s"%(driver,server,db,uid,pwd)
    conn_str = "DSN=%s;UID=%s;PWD=%s"%('MSSQL',uid,pwd)
    print conn_str
    conn = pyodbc.connect(conn_str)
    cur = conn.cursor()
    q = 'SELECT * FROM %s ORDER BY TIMESTAMP asc'%(table)
    cur.execute(q)
    rows = cur.fetchall()
    return rows

def read_steam_csv(**keywords):
    '''
    args:
        fmt: Date format
        filename: dir path + filename of input file
    '''
    if 'fmt' not in keywords: fmt="%Y-%m-%d %H:%M:%S.%f"
    else: fmt = keywords['fmt']
    if 'filename' not in keywords: filename="../Data/RUDINSERVER_CURRENT_STEAM_DEMAND_FX70_AUG.csv"
    else: filename = keywords['filename']
    f = open(filename,'rU')
    steam_data = []
    cnt = 0
    for line in f:
        vals = line.rstrip().split(',')
        try:
            vals[1] = vals[1][:23] #trim microseconds to 3 digits
            steam_date = dt.datetime.strptime(vals[1],fmt) 
            steam_data.append((steam_date,float(vals[2])))
        except:
            print vals[1],'val: ',vals[2]
            print "INCORRECT DATETIME FORMAT"
            cnt += 1
    f.close()
    print 'Count: %d'%(cnt)
    return steam_data

def read_weather_csv(**keywords):
    '''
    args:
    fmt = date format
    filename = csv filename for weather data
    '''
    print keywords
    if 'fmt' not in keywords: fmt="%Y-%m-%d %H:%M:%S.%f"
    else: fmt = keywords['fmt']
    if 'filename' not in keywords: filename="../Data/OBSERVED_WEATHER_AUG.csv"
    else: filename = keywords['filename']
    f = open(filename,'rU')
    weather_data = []
    for line in f:
        vals = line.rstrip().split(',')
        try:
            vals[1] = vals[1][:23] #trim microseconds to 3 digits
            weather_date = dt.datetime.strptime(vals[1],fmt) 
            weather_data.append((weather_date,vals[2],vals[4]))
        except:
            print "INCORRECT DATETIME FORMAT"
    f.close()
    weather_data.reverse() #change the sorted order to asc
    return weather_data

def read_steam_data(**keywords):
    '''
    args:
    fmt = date format
    filename = csv filename for steam data
    '''
    return read_steam_csv(**keywords)

def read_weather_data(**keywords):
    '''
    args:
    fmt = date format
    filename = csv filename for weather data
    '''
    return read_weather_csv(**keywords)


def main():
    return read_db()

if __name__=='__main__':
    main()
