from flask import Flask
from flask import request
from flask import render_template
from werkzeug import secure_filename
import os
app = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app

@app.route('/',methods=['GET','POST'],endpoint = 'hello')
def hello():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html')
        else:
            file.save(secure_filename(file.filename))
            analyze(file.filename)
            return render_template('output.html')
    else:
        return "Invalid inputs"
def analyze(file_name):
    
    
    import pandas as pd
    
    import numpy as np 
    
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    print("done downloading")
    
    import matplotlib.pyplot as plt
    
    
    
    import re
    
    from textblob import TextBlob
    from textblob import download_corpora

    
   
    """
    Importing the file
    
    """
    #read data "Final_data.csv"
    df = pd.read_csv(file_name,encoding='latin1')
    df['createdon'] = pd.to_datetime(df['createdon'])
    
    filter_division = ["Membership Sales - IS","Membership Sales - AE","Group Sales - Sports","Group Sales - Entertainment","Suite Services","Suite Sales" ]
    df = df.loc[df['msg_departmentname'].isin(filter_division)]
    
    #sorting by time and contact id
    df = df.sort_values(['contactid', 'createdon' ], ascending=[True, True])
    df = df.reset_index(drop=True)
    
    
    
    """
    Creating labels for response
    
    """
    df['Time_Diff'] = 0
    df['Conv_Start'] = 0
    df.loc[0,'Conv_Start'] = 1
    df['Text_id'] = 0
    df.loc[0,'Text_id'] = 1
    
    df['To_keep'] = 0
    #df1.loc[0,'To_keep'] = 1
    if (df.loc[0,'Text_id'] == 1 and df.loc[0,'new_inbound'] == 0 ) : df.loc[0,'To_keep'] = 1
    elif (df.loc[0,'Text_id'] == 2 and df.loc[0,'new_inbound'] == 1 ) : df.loc[0,'To_keep'] = 1
    else : df.loc[0,'To_keep'] = 0
    
    df['resp'] = 0
    
    
    df['length'] = 0
    
    for i in range(1, len(df)-1):
      #df.loc[i,'length'] = len(str(df.loc[i,'description'])) 
      time1  = df.loc[i,'createdon']
      time2  = df.loc[i-1,'createdon']
      
      diff = time1 - time2
      dif_in_hours = diff.days * 24 + (diff.seconds) / 3600
      
      df.loc[i,'Time_Diff'] = dif_in_hours
      
      if df.loc[i,'contactid'] == df.loc[i-1,'contactid']: df.loc[i,'Conv_Start'] = 0
      else : df.loc[i,'Conv_Start'] = 1
        
      if df.loc[i,'contactid'] == 1 : df.loc[i,'Text_id'] = 1
      elif df.loc[i,'Time_Diff'] <= -0.01 or df.loc[i,'Time_Diff'] >= 12.1  : df.loc[i,'Text_id'] = 1
      else : df.loc[i,'Text_id'] = df.loc[i-1,'Text_id'] + 1
      
      if (df.loc[i,'Text_id'] == 1 and df.loc[i,'new_inbound'] == 0 ) : df.loc[i,'To_keep'] = 1
      elif (df.loc[i,'Text_id'] == 2 and df.loc[i,'new_inbound'] == 1 and df.loc[i,'new_inbound'] != df.loc[i-1,'new_inbound']  ) : df.loc[i,'To_keep'] = 1
      else : df.loc[i,'To_keep'] = 0
      #if(i%1000 == 0 ): print(datetime.now())
        
    
    #print(df.head())
    print("end p1")
    
    """
    Replacing rep and clients in the text
    
    """   
    
    for i in range(len(df)-1):
      if (df.loc[i,'Text_id'] == 1 and df.loc[i,'new_inbound'] == 0 and df.loc[i+1,'Text_id'] == 2 and df.loc[i+1,'new_inbound'] == 1 ) :df.loc[i,'resp'] = 1
      st = df.loc[i,"description"]
      if (df.loc[i,"To_keep"]) == 1:
        
        try:
           
          rep = df.loc[i,"owneridname"]
          client = df.loc[i,"regardingobjectidname"]
          st = str(df.loc[i,"description"])
          st_split = re.sub('[^A-Za-z0-9\s]','',st).split()
          
          df.loc[i,'length'] = len(str(st_split))
          
          
          for a in range(len(st_split)-1):
            
            rep_split = rep.split() 
            client_split = client.split()
            
            comb = st_split[a] +" "+ st_split[a + 1]
            
            
              
            if st_split[a].lower() == rep_split[0].lower() or  st_split[a].lower() == rep_split[1].lower() or st_split[a].lower() in ["mike","jen","chris","demi"]:
              st = st.replace(st_split[a],"REP")
              
              
            elif st_split[a].lower() == client_split[0].lower() or  st_split[a].lower() == client_split[1].lower():
              st = st.replace(st_split[a],"CLIENT")
              
      
        except Exception as ex:
            #print(ex)
          continue
          
      df.loc[i,"description"] = st
        
    df.to_csv ('data.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
    
    print("end p2")
    
    
    """
    df_toKeep : Data with all the inbounds and outbounds tha need to be kept according to the rules:
      1. The first time a conversation starts with an outbound message (OUTBOUND)
      2. A reply is recieved in 24 hours (INBOUND)
    
    df3 : only the outbound messages are kept which received a response
    
    """ 
    #set the date and time format
    #date_format = "%m/%d/%Y %H:%M"
    
    df = df.sort_values(['contactid', 'createdon' ], ascending=[True, True])
    df = df.reset_index(drop=True)
    
    df_toKeep = df[(df['To_keep'] == 1  ) ]
    df_toKeep = df_toKeep.reset_index()
    
    
    
    #Creating the plot for rep engagement rate
    #sentiment analysis of answers and giving the scores to their questions
    
    df_sentiment = df_toKeep
    df_sentiment['polarity'] = 0
    df_sentiment['sa_score'] = 0
    df_sentiment = df_sentiment.sort_index(ascending=False)
    df_sentiment = df_sentiment.reset_index(drop=False)
    
    df_sentiment['sentiment'] = 'x'
    
    
    for descr in range(0,len(df_sentiment['description'])):
      sa_score = 0
      try:
        x = (TextBlob(df_sentiment['description'][descr]).sentiment)
        
        if x[0] <= -0.1 : sa_score = -1
        elif x[0] >= 0.05 : sa_score = 1
        else : sa_score = 0
    
        df_sentiment.loc[descr,'polarity'] = x[0]
        df_sentiment.loc[descr,'sa_score'] = sa_score
        
        
      except:
        df_sentiment.loc[descr,'polarity'] = 0
          
          
          
      if descr == 0: continue
      else:
        if df_sentiment['new_inbound'][descr] == 0 and df_sentiment['new_inbound'][descr-1] == 1:
          if df_sentiment['polarity'][descr-1] <= -0.1 : df_sentiment.loc[descr,'sentiment'] = -1
          elif df_sentiment['polarity'][descr-1] >= 0.05 : df_sentiment.loc[descr,'sentiment'] = 1
          else : df_sentiment.loc[descr,'sentiment'] = 0
          
    df_sentiment = df_sentiment.sort_index(ascending=False)
    df_sentiment = df_sentiment.reset_index(drop=True)
          
    #print(df_sentiment.head())
    filename = file_name+"sentiment.csv"
    
    df_sentiment.to_csv (filename, index = None, header=True) 
    print("done")
    
    print("\n% Outbound by Sentiments")  
    df_sentiment_outbounds = df_sentiment[(df_sentiment['new_inbound'] == 0)]
    
    
    df_sentiment_outbounds = df_sentiment_outbounds.reset_index(drop = True)
    sa_eng = [0,0,0]
    sa_prob_senti = [0,0,0]
    #sa_len_total = [0,0,0,0]
    #sa_prob_len = [0,0,0,0]
    
    for senti in range(len(df_sentiment_outbounds['sa_score'])):
      if df_sentiment_outbounds.loc[senti,'sa_score' ]== 1:
        if  df_sentiment_outbounds.loc[senti,'resp' ]== 1:
          sa_prob_senti[0] +=1
        sa_eng[0] += 1
        
      elif df_sentiment_outbounds.loc[senti,'sa_score' ]== 0:
        if  df_sentiment_outbounds.loc[senti,'resp' ]== 1:
          sa_prob_senti[1] +=1
        sa_eng[1] += 1
        
      else:
        if  df_sentiment_outbounds.loc[senti,'resp' ]== 1:
          sa_prob_senti[2] +=1
        sa_eng[2] += 1
      
    sentiment_word = ["Positive","Neutral","Negative"]
    
    for i in range(3):
      print(sentiment_word[i]," : " + str(sa_eng[i])+ " Outbound Messages")
    
      print("Probability(Response |",sentiment_word[i],"Outbound Message) :",str(round(100*sa_prob_senti[i]/sa_eng[i],2)),"\n")
    
    import numpy as np
    import pandas as pd
    
    
    in_senti_list = [-1,0,1]
    out_senti_list = [-1,0,1]
    
    df_try = pd.DataFrame(0, index=out_senti_list, columns=in_senti_list)
    df_try_percent = pd.DataFrame(0, index=out_senti_list, columns=in_senti_list)
    
    
    for senti in range(len(df_sentiment_outbounds['sentiment'])):
      if df_sentiment_outbounds.loc[senti,'sentiment'] != 'x':
        df_try.loc[df_sentiment_outbounds.loc[senti,'sa_score'],df_sentiment_outbounds.loc[senti,'sentiment']] += 1
    print(df_try)
    
    sum_resp = df_try.sum(axis=1)
    print(sum_resp)
    
    for i in out_senti_list:
      for j in in_senti_list:
        df_try_percent.loc[i,j] = int(df_try.loc[i,j]*100/sum_resp[i])
    print(df_try_percent)
    
    #export_csv = df_try_percent.to_csv ('Seniment_CM.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15,6))
    
    # set width of bar
    barWidth = 0.2
     
    # set height of bar
    bars1 = df_try_percent[-1]
    bars2 = df_try_percent[0]
    bars3 = df_try_percent[1]
     
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
     
    # Make the plot
    plt.bar(r1, bars1, color='#217BFE', width=barWidth/1.25, edgecolor='white', label='Negative In')
    plt.bar(r2, bars2, color='#FE8221', width=barWidth/1.25, edgecolor='white', label='Neutral In')
    plt.bar(r3, bars3, color='#AAB0BA', width=barWidth/1.25, edgecolor='white', label='Positive In')
     
    # Add xticks on the middle of the group bars
    plt.xlabel('Sentiment Of Outbound Messages', fontweight='bold', fontsize = 16)
    plt.ylabel('% of Inbound Messages', fontweight='bold', fontsize = 16)
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Negative Out', 'Neutral Out', 'Positive Out'], fontsize = 14)
    
    plt.suptitle('Sentiment Matrix', fontsize=20)
    
    # Create legend & Show graphic
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3)
    cwd = os.getcwd()+'\static\photo\plot.png'
    plt.savefig(cwd)
   
    print("\n% Replies by Sentiments") 
    df_sentiment_eng = df_sentiment[(df_sentiment['sentiment']!='x')]
    #print(df_sentiment_eng.head(15))
    sa_eng = [0,0,0]
    for senti in df_sentiment_eng['sentiment']:
      if senti == 1:
        sa_eng[0] += 1
      elif senti == 0:
        sa_eng[1] += 1
      else:
        sa_eng[2] += 1
        
    sentiment_word = ["Positive : ","Neutral : ","Negative : "]
    
    for i in range(3):
      print(sentiment_word[i] + str(round(100*sa_eng[i]/len(df_sentiment_eng['sentiment']),2)))

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
