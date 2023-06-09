from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import io
import os
import base64
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fileinput import filename
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from apyori import apriori as apri
from scipy.spatial.distance import cdist    
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
plt.style.use('ggplot')


app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'database'

mysql = MySQL(app)


UPLOAD_FOLDER = 'app/static/files/'



# Paginas principales

@app.route('/')
def index(): 
    cursos = ['PHP','C++','Java']
    data = {
        'titulo':'InvestAI | La inteligencia de tus inversiones',
    }
    return render_template('index.html', data=data)

@app.route('/contact')
def contacto():
    data = {
        'titulo':'Contacto',
    }
    return render_template('contact.html', data=data)

@app.route('/addcontact', methods=['POST'])
def addcontact():
    if request.method =='POST' :
        fullname = request.form['fullname']
        phone = request.form['phone']
        email = request.form['email']
        message = request.form['message']
        cur = mysql.connection.cursor()
        cur.execute('INSERT INTO contacts (fullname, phone, email, message) VALUES (%s, %s, %s, %s)',(fullname, phone, email, message))
        mysql.connection.commit()
        return render_template('thanks.html')
    


@app.route('/getapri', methods=['GET','POST'])
def getapri():
    data = {
        'titulo':'Reglas de asociación',
    }

    if request.method =='POST' :
        selectapri = request.form['selectapri']
        
        if(selectapri == "1"):

            f = request.files['csvdata']
            f.save(UPLOAD_FOLDER+"apridata.csv")
            
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"apridata.csv")
            Transacciones = DatosMovies.values.reshape(-1).tolist()
            Lista = pd.DataFrame(Transacciones)
            Lista['Frecuencia'] = 1
            Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
            Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
            Lista = Lista.rename(columns={0 : 'Item'})
            Lista = Lista[ Lista.Item.str.len()>0]
           
            plt.figure(figsize=(16,20))
            plt.title('Grafica de frecuencias')
            plt.ylabel('Item')
            plt.xlabel('Porcentaje de frecuencias')
            plt.barh(Lista['Item'], width=Lista['Porcentaje'], color='#0099ff')
            plt.savefig('app/static/plots/freqplot.png', dpi=250)


        

        elif(selectapri == "2"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM movies')
            df = cur.fetchall()
           
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"apridata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"apridata.csv")

            Transacciones = DatosMovies.values.reshape(-1).tolist()
            Lista = pd.DataFrame(Transacciones)
            Lista['Frecuencia'] = 1
            Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
            Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
            Lista = Lista.rename(columns={0 : 'Item'})
            Lista = Lista[ Lista.Item.str.len()>0]
           
            plt.figure(figsize=(16,20))
            plt.title('Grafica de frecuencias')
            plt.ylabel('Item')
            plt.xlabel('Porcentaje de frecuencias')
            plt.barh(Lista['Item'], width=Lista['Porcentaje'], color='#0099ff')
            plt.savefig('app/static/plots/freqplot.png', dpi=250)



            
            
        return render_template('getapri.html', plot_url ='static/plots/freqplot.png', data=data)
       
    return render_template('apriori.html')


@app.route('/showapri', methods=['GET','POST'])
def showapri():
    data = {
        'titulo':'Reglas de asociación',
    }

    if request.method =='POST' :
        support = float(request.form['support'])
        confidence = float(request.form['confidence'])
        lift = float(request.form['lift'])
        

        DatosMovies = pd.read_csv(UPLOAD_FOLDER+"apridata.csv")
        MoviesLista = DatosMovies.stack().groupby(level=0).apply(list).tolist()
        ReglasC1 = apri(MoviesLista, min_support=support, min_confidence=confidence, min_lift=lift)
        ResultadosC1 = list(ReglasC1)
        headings = ["Regla","Soporte","Confianza","Elevación"]
        table=[]
        for i in range(len(ResultadosC1)):
            table.append([])

        j=0
        for item in ResultadosC1:
            
            #El primer índice de la lista
            new = str(item[0]).replace("frozenset({","")

            table[j].append(new.replace("})",""))

            #El segundo índice de la lista
            table[j].append(str(item[1]))

            #El tercer índice de la lista
            table[j].append(str(item[2][0][2]))
            table[j].append(str(item[2][0][3]))
            j+=1
        print(table)



    return render_template('showapri.html', data=data, table=table, headings=headings)


@app.route('/getdist', methods=['GET','POST'])
def getdist():
    data = {
        'titulo':'Métricas de distancia',
    }

    if request.method =='POST' :
        selectdist = request.form['selectdist']
        
        if(selectdist == "1"):

            f = request.files['csvdata']
            f.save(UPLOAD_FOLDER+"distdata.csv")
            
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"distdata.csv")

        

        elif(selectdist == "2"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM hipoteca')
            df = cur.fetchall()
           
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"distdata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"distdata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"distdata.csv", index=False)



        elif(selectdist == "3"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM diabetes')
            df = cur.fetchall()
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"distdata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"distdata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"distdata.csv", index=False)
                
        return render_template('getdist.html', data=data)
        
    return render_template('distance.html')



@app.route('/showdist', methods=['GET','POST'])
def showdist():
    data = {
        'titulo':'Métricas de distancia',
    }

    if request.method =='POST' :
        esc = float(request.form['selectest'])
        dist = float(request.form['selectdist'])
        
        

        df = pd.read_csv(UPLOAD_FOLDER+"distdata.csv")
  
        print(esc)
        if(esc == 0):
            stand = MinMaxScaler()
        elif(esc==1):
            stand = StandardScaler()

        MEstandarizada = stand.fit_transform(df)
        MEstandarizada = pd.DataFrame(MEstandarizada)
        
        if(dist==0):
            DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
            Matrix = pd.DataFrame(DstEuclidiana)
            

        elif(dist==1):
            DstChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
            Matrix = pd.DataFrame(DstChebyshev)
            

        elif(dist==2):
            DstManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
            Matrix = pd.DataFrame(DstManhattan)
            

        elif(dist==3):
            DstMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=1.5)
            Matrix = pd.DataFrame(DstMinkowski)
            
    

        table = Matrix.to_records().tolist()

        return render_template('showdist.html', data=data, table=table)






@app.route('/getclustjer', methods=['GET','POST'])
def getclustjer():
    data = {
        'titulo':'Clustering Jerárquico',
    }

    if request.method =='POST' :
        selectclust = request.form['selectclust']
        
        if(selectclust == "1"):

            f = request.files['csvdata']
            f.save(UPLOAD_FOLDER+"clustdata.csv")
            
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"clustdata.csv")


  

        

        elif(selectclust == "2"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM hipoteca')
            df = cur.fetchall()
           
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"clustdata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"clustdata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            DatosMovies.drop(['comprar'], axis=1)
            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"clustdata.csv", index=False)
   



        elif(selectclust == "3"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM diabetes')
            df = cur.fetchall()
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"clustdata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"clustdata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            DatosMovies = DatosMovies.drop(['Outcome'], axis=1)

            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"clustdata.csv", index=False)
        return render_template('getclustjer.html', data=data)
        
    return render_template('clusteringjer.html')



@app.route('/showjerclust', methods=['GET','POST'])
def showjerclust():
    data = {
        'titulo':'Clustering Jerárquico',
    }

    if request.method =='POST' :
        esc = float(request.form['selectest'])

        typedist = float(request.form['typedist'])
        
        

        df = pd.read_csv(UPLOAD_FOLDER+"clustdata.csv")

        if(esc == 0):
            stand = MinMaxScaler()
        elif(esc==1):
            stand = StandardScaler()

        MEstandarizada = stand.fit_transform(df)
        MEstandarizada = pd.DataFrame(MEstandarizada)
        pd.DataFrame(MEstandarizada).to_csv(UPLOAD_FOLDER+"jerest.csv", index=False)

      
        plt.figure(figsize=(10, 7))
        plt.ylabel('Distancia')
        if(typedist==0):
            Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
            plt.savefig('app/static/plots/jerplot.png', dpi=250)  
        elif(typedist==1):
            Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='chebyshev'))
            plt.savefig('app/static/plots/jerplot.png', dpi=250)
        elif(typedist==2):
            Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='cityblock'))
            plt.savefig('app/static/plots/jerplot.png', dpi=250)
            
    


        return render_template('showjerclust.html', data=data, plot_url ='static/plots/jerplot.png')
    return render_template('getclustjer.html', data=data, plot_url ='static/plots/jerplot.png')




@app.route('/finaljerclust', methods=['GET','POST'])
def finaljerclust():
    data = {
        'titulo':'Clustering Jerárquico',
    }

    if request.method =='POST' :
        num = int(request.form['num'])

        MEstandarizada = pd.read_csv(UPLOAD_FOLDER+"jerest.csv")
        
        MJerarquico = AgglomerativeClustering(n_clusters=num, linkage='complete', affinity='euclidean')
        MJerarquico.fit_predict(MEstandarizada)
        DatosMovies = pd.read_csv(UPLOAD_FOLDER+"clustdata.csv")
        DatosMovies['cluster'] = MJerarquico.labels_

        table = DatosMovies.to_records().tolist()
        headings = DatosMovies.columns
        headings = headings.insert(0, 'index')

        Centroides = DatosMovies.groupby('cluster').mean()

        table2 = Centroides.to_records().tolist()
        headings2 = DatosMovies.columns
        headings2 = headings2.insert(0, 'cluster')
        headings2 = headings2[:-1]
        
            
    


        return render_template('finaljerclust.html', data=data, table=table, headings=headings, table2=table2, headings2=headings2)
  


@app.route('/getclustpar', methods=['GET','POST'])
def getclustpar():
    data = {
        'titulo':'Clustering particional',
    }

    if request.method =='POST' :
        selectdist = request.form['selectclust']
        
        if(selectdist == "1"):

            f = request.files['csvdata']
            f.save(UPLOAD_FOLDER+"parclustdata.csv")
            
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"parclustdata.csv")
  
        

        elif(selectdist == "2"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM hipoteca')
            df = cur.fetchall()
           
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"parclustdata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"parclustdata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"parclustdata.csv", index=False)
   


        elif(selectdist == "3"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM diabetes')
            df = cur.fetchall()
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"parclustdata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"parclustdata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"parclustdata.csv", index=False)

        df = pd.read_csv(UPLOAD_FOLDER+"parclustdata.csv")
  
        stand = MinMaxScaler()

        MEstandarizada = stand.fit_transform(df)
        MEstandarizada = pd.DataFrame(MEstandarizada)
        SSE = []
        for i in range(2, 12):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(MEstandarizada)
            SSE.append(km.inertia_)

        #Se grafica SSE en función de k
        plt.figure(figsize=(10, 7))
        plt.plot(range(2, 12), SSE, marker='o')
        plt.xlabel('Cantidad de clusters *k*')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")  
        kl.plot_knee() 
        plt.savefig('app/static/plots/parplot.png',dpi=250)   
                
        return render_template('getclustpar.html', data=data, plot_url ='static/plots/parplot.png')
        
    return render_template('clusteringpar.html')



@app.route('/showclustpar', methods=['GET','POST'])
def showclustpar():
    data = {
        'titulo':'Cluster Particional',
    }

    if request.method =='POST' :
        std = int(request.form["selectest"])
        num = int(request.form['num'])

        df = pd.read_csv(UPLOAD_FOLDER+"parclustdata.csv")

        if(std == 0):
            stand = MinMaxScaler()
        elif(std == 1): 
            stand = StandardScaler()

        MEstandarizada = stand.fit_transform(df)
        MEstandarizada = pd.DataFrame(MEstandarizada)
  
 
        
        MParticional = KMeans(n_clusters=num, random_state=0).fit(MEstandarizada)
        MParticional.predict(MEstandarizada)

        DatosMovies = pd.read_csv(UPLOAD_FOLDER+"parclustdata.csv")
        DatosMovies['cluster'] = MParticional.labels_

        table = DatosMovies.to_records().tolist()
        headings = DatosMovies.columns
        headings = headings.insert(0, 'index')

        Centroides = DatosMovies.groupby('cluster').mean()

        table2 = Centroides.to_records().tolist()
        headings2 = DatosMovies.columns
        headings2 = headings2.insert(0, 'cluster')
        headings2 = headings2[:-1]



        return render_template('showclustpar.html', data=data, table=table, headings=headings, table2=table2, headings2=headings2)




@app.route('/getlogclas', methods=['GET','POST'])
def getlogclas():
    data = {
        'titulo':'Clasificación Logística',
    }

    if request.method =='POST' :
        selectdist = request.form['selectclust']
        
        if(selectdist == "1"):

            f = request.files['csvdata']
            f.save(UPLOAD_FOLDER+"logdata.csv")
            
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"logdata.csv")
  
        

        elif(selectdist == "2"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM hipoteca')
            df = cur.fetchall()
           
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"logdata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"logdata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"logdata.csv", index=False)
   


        elif(selectdist == "3"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM diabetes')
            df = cur.fetchall()
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"logdata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"logdata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"logdata.csv", index=False)

        df = pd.read_csv(UPLOAD_FOLDER+"logdata.csv")
        table = df.to_records().tolist()
        headings = df.columns
        headings = headings.insert(0, 'index')

       
        return render_template('getlogclas.html', data=data, table=table, headings=headings)
        
    return render_template('logclas.html')




@app.route('/showlogclas', methods=['GET','POST'])
def showlogclas():
    data = {
        'titulo':'Clasificación Logística',
    }

    if request.method =='POST' :
        vari = str(request.form["vari"])


        df = pd.read_csv(UPLOAD_FOLDER+"logdata.csv")

        Y = df[vari]
        Y = np.array(Y)

        X = df.drop(vari, axis=1)
        X = np.array(X)
       

        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                test_size = 0.2, 
                                                                random_state = 1234,
                                                                shuffle = True)
        
        global ClasificacionRL
        ClasificacionRL = linear_model.LogisticRegression()
        ClasificacionRL.fit(X_train, Y_train)
        Probabilidad = ClasificacionRL.predict_proba(X_validation)
        Y_ClasificacionRL = ClasificacionRL.predict(X_validation)
        acc = accuracy_score(Y_validation, Y_ClasificacionRL)
        acc = acc*100

        ModeloClasificacion = ClasificacionRL.predict(X_validation)
        Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                        ModeloClasificacion, 
                                        rownames=['Reales'], 
                                        colnames=['Clasificación'])
        clasmat = pd.DataFrame(Matriz_Clasificacion)
        clasrep = classification_report(Y_validation, Y_ClasificacionRL)
        CurvaROC = RocCurveDisplay.from_estimator(ClasificacionRL, X_validation, Y_validation)

      
        headings = pd.DataFrame({'col1': ["Clasificación", "Reales"], 'col2': [0, ""], 'col3': [1, ""]})
        headings = headings.to_records(index=False).tolist()

        table = clasmat.to_records().tolist()
        plt.savefig('app/static/plots/logplot.png',dpi=250)
 
        
        datos = df.drop(vari, axis=1)
        datos = datos.columns
        length = len(datos)
        pd.DataFrame(datos).to_csv(UPLOAD_FOLDER+"vardat.csv", index=False)
        
        


        return render_template('showlogclas.html', data=data, acc=acc, table=table, headings=headings, 
                               plot_url ='static/plots/logplot.png', datos=datos, length=length)
    

@app.route('/finalclas', methods=['GET','POST'])
def finalclas():
    data = {
        'titulo':'Clasificación Logística',
    }

    if request.method =='POST' :

        df1 = pd.read_csv(UPLOAD_FOLDER+"vardat.csv")
        datos = df1.iloc[:,0].tolist()
        

        inputs = {}
        df = pd.read_csv(UPLOAD_FOLDER+"formul.csv")
        formul = df.to_records().tolist()
        formul = formul[0][1:]
 

        for i in range(len(datos)):
            inputs[datos[i]] =[float(request.form[datos[i]])]

        inputs = pd.DataFrame(inputs)
        

      
        pred =ClasificacionRL.predict(inputs)
        pred = pred[0]

        



        return render_template('finalclas.html', data=data, pred=pred)



@app.route('/getregtree', methods=['GET','POST'])
def getregtree():
    data = {
        'titulo':'Árboles Aleatorios',
    }

    if request.method =='POST' :
        selectdist = request.form['selectclust']
        
        if(selectdist == "1"):

            f = request.files['csvdata']
            f.save(UPLOAD_FOLDER+"treedata.csv")
            
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"treedata.csv")
  
        

        elif(selectdist == "2"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM hipoteca')
            df = cur.fetchall()
           
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"treedata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"treedata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"treedata.csv", index=False)
   


        elif(selectdist == "3"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM diabetes')
            df = cur.fetchall()
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"treedata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"treedata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"treedata.csv", index=False)

        df = pd.read_csv(UPLOAD_FOLDER+"treedata.csv")
        table = df.to_records().tolist()
        headings = df.columns
        headings = headings.insert(0, 'index')

       
        return render_template('getregtree.html', data=data, table=table, headings=headings)
        
    return render_template('regtree.html')



@app.route('/showtree', methods=['GET','POST'])
def showtree():
    data = {
        'titulo':'Árboles Aleatorios',
    }

    if request.method =='POST' :
        vari = str(request.form["vari"])


        df = pd.read_csv(UPLOAD_FOLDER+"treedata.csv")

        Y = df[vari]
        Y = np.array(Y)

        X = df.drop(vari, axis=1)
        X = np.array(X)
       

        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                test_size = 0.2, 
                                                                random_state = 1234,
                                                                shuffle = True)
        
        global ClasificacionAD
        ClasificacionAD = DecisionTreeClassifier(random_state=0)
        ClasificacionAD.fit(X_train, Y_train)
        Probabilidad = ClasificacionAD.predict_proba(X_validation)
        Y_ClasificacionAD = ClasificacionAD.predict(X_validation)


        acc = ClasificacionAD.score(X_validation, Y_validation)
        acc = acc*100

        Y_ClasificacionAD = ClasificacionAD.predict(X_validation)
        Matriz_ClasificacionAD = pd.crosstab(Y_validation.ravel(), 
                                        Y_ClasificacionAD, 
                                        rownames=['Real'], 
                                        colnames=['Clasificación'])
        clasmat = pd.DataFrame(Matriz_ClasificacionAD)

        clasrep = classification_report(Y_validation, Y_ClasificacionAD)
        ndf = df.drop(vari, axis=1)
        Importancia = pd.DataFrame({'Variable': list(ndf),
                            'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)


        

        headings = pd.DataFrame({'col1': ["Clasificación", "Reales"], 'col2': [0, ""], 'col3': [1, ""]})
        headings = headings.to_records(index=False).tolist()

        table = clasmat.to_records().tolist()


        plt.figure(figsize=(16,16))  

        plot_tree(ClasificacionAD,)
        plt.savefig('app/static/plots/treeplot.png',dpi=400)

        headings2 = Importancia.columns
        headings2 = headings2.insert(0, 'index')
        table2 = Importancia.to_records().tolist()

        table = clasmat.to_records().tolist()

        
        datos = df.drop(vari, axis=1)
        datos = datos.columns
        length = len(datos)
        pd.DataFrame(datos).to_csv(UPLOAD_FOLDER+"vardat.csv", index=False)
        
        reporte = export_text(ClasificacionAD, 
                      feature_names = (ndf.columns).tolist())
        reporte = reporte.replace(" ", ".")
        reporte = reporte.split("\n")
        

        return render_template('showtree.html', data=data, acc=acc, table=table, headings=headings,table2=table2, headings2=headings2, 
                               plot_url ='static/plots/treeplot.png', datos=datos, length=length, reporte=reporte)
    

@app.route('/finaltree', methods=['GET','POST'])
def finaltree():
    data = {
        'titulo':'Árboles Aleatorios',
    }

    if request.method =='POST' :

        df1 = pd.read_csv(UPLOAD_FOLDER+"vardat.csv")
        datos = df1.iloc[:,0].tolist()
        

        inputs = {}
        

        


        for i in range(len(datos)):
            inputs[datos[i]] =[float(request.form[datos[i]])]

        inputs = pd.DataFrame(inputs)
        print(inputs)
    
        

      
        pred =ClasificacionAD.predict(inputs)
        pred = pred[0]

        



        return render_template('finaltree.html', data=data, pred=pred)









@app.route('/getforest', methods=['GET','POST'])
def getforest():
    data = {
        'titulo':'Bosques Aleatorios',
    }

    if request.method =='POST' :
        selectdist = request.form['selectclust']
        
        if(selectdist == "1"):

            f = request.files['csvdata']
            f.save(UPLOAD_FOLDER+"fordata.csv")
            
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"fordata.csv")
  
        

        elif(selectdist == "2"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM hipoteca')
            df = cur.fetchall()
           
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"fordata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"fordata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"fordata.csv", index=False)
   


        elif(selectdist == "3"):
            cur = mysql.connection.cursor()
            cur.execute('SELECT * FROM diabetes')
            df = cur.fetchall()
            pd.DataFrame(df).to_csv(UPLOAD_FOLDER+"fordata.csv", index=False)
            DatosMovies = pd.read_csv(UPLOAD_FOLDER+"fordata.csv")
            DatosMovies.columns = DatosMovies.iloc[0]
            DatosMovies = DatosMovies[1:]
            pd.DataFrame(DatosMovies).to_csv(UPLOAD_FOLDER+"fordata.csv", index=False)

        df = pd.read_csv(UPLOAD_FOLDER+"fordata.csv")
        table = df.to_records().tolist()
        headings = df.columns
        headings = headings.insert(0, 'index')

       
        return render_template('getforest.html', data=data, table=table, headings=headings)
        
    return render_template('forest.html')



@app.route('/showforest', methods=['GET','POST'])
def showforest():
    data = {
        'titulo':'Bosques Aleatorios',
    }

    if request.method =='POST' :
        vari = str(request.form["vari"])


        df = pd.read_csv(UPLOAD_FOLDER+"fordata.csv")

        Y = df[vari]
        Y = np.array(Y)

        X = df.drop(vari, axis=1)
        X = np.array(X)
       

        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                test_size = 0.2, 
                                                                random_state = 1234,
                                                                shuffle = True)
        
        global ClasificacionBA
        ClasificacionBA = RandomForestClassifier(random_state=0)
        ClasificacionBA.fit(X_train, Y_train)
        Probabilidad = ClasificacionBA.predict_proba(X_validation)
        Y_ClasificacionBA = ClasificacionBA.predict(X_validation)


        acc = ClasificacionBA.score(X_validation, Y_validation)
        acc = acc*100

        Y_ClasificacionBA = ClasificacionBA.predict(X_validation)
        Matriz_ClasificacionBA = pd.crosstab(Y_validation.ravel(), 
                                        Y_ClasificacionBA, 
                                        rownames=['Real'], 
                                        colnames=['Clasificación'])
        clasmat = pd.DataFrame(Matriz_ClasificacionBA)

        clasrep = classification_report(Y_validation, Y_ClasificacionBA)
        ndf = df.drop(vari, axis=1)
        Importancia = pd.DataFrame({'Variable': list(ndf),
                            'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)


        

        headings = pd.DataFrame({'col1': ["Clasificación", "Reales"], 'col2': [0, ""], 'col3': [1, ""]})
        headings = headings.to_records(index=False).tolist()

        table = clasmat.to_records().tolist()



        headings2 = Importancia.columns
        headings2 = headings2.insert(0, 'index')
        table2 = Importancia.to_records().tolist()

        table = clasmat.to_records().tolist()

        
        datos = df.drop(vari, axis=1)
        datos = datos.columns
        length = len(datos)
        pd.DataFrame(datos).to_csv(UPLOAD_FOLDER+"vardat.csv", index=False)
        

        

        return render_template('showforest.html', data=data, acc=acc, table=table, headings=headings,table2=table2, headings2=headings2, 
                               plot_url ='static/plots/treeplot.png', datos=datos, length=length)
    

@app.route('/finalforest', methods=['GET','POST'])
def finalforest():
    data = {
        'titulo':'Bosques Aleatorios',
    }

    if request.method =='POST' :

        df1 = pd.read_csv(UPLOAD_FOLDER+"vardat.csv")
        datos = df1.iloc[:,0].tolist()
        

        inputs = {}
        

        


        for i in range(len(datos)):
            inputs[datos[i]] =[float(request.form[datos[i]])]

        inputs = pd.DataFrame(inputs)
        print(inputs)
    
        

      
        pred =ClasificacionBA.predict(inputs)
        pred = pred[0]

        



        return render_template('finalforest.html', data=data, pred=pred)











@app.route('/services')
def services():
    data = {
        'titulo':'Servicios',
    }
    return render_template('services.html', data=data)

@app.route('/about')
def about():
    data = {
        'titulo':'Acerca de',
    }
    return render_template('about.html', data=data)



# Algoritmos

@app.route('/apriori')
def apriori():
    data = {
        'titulo':'Reglas de Asociación',
    }
    return render_template('apriori.html', data=data)

@app.route('/distance')
def distance():
    data = {
        'titulo':'Métricas de distancia',
    }
    return render_template('distance.html', data=data)

@app.route('/clusteringjer')
def clusteringjer():
    data = {
        'titulo':'Clustering Jerárquico',
    }
    return render_template('clusteringjer.html', data=data)

@app.route('/clusteringpar')
def clusteringpar():
    data = {
        'titulo':'Clustering Particional',
    }
    return render_template('clusteringpar.html', data=data)

@app.route('/logclas')
def logclas():
    data = {
        'titulo':'Clasificación Logística',
    }
    return render_template('logclas.html', data=data)



@app.route('/regtree')
def regtree():
    data = {
        'titulo':'Árboles Aleatorios',
    }
    return render_template('regtree.html', data=data)

@app.route('/forest')
def linearreg():
    data = {
        'titulo':'Bosque Aleatorio',
    }
    return render_template('forest.html', data=data)


#Error 404

def pagina_no_encontrada(error):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.register_error_handler(404, pagina_no_encontrada)
    app.run(debug=True)