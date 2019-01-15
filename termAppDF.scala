import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import sqlContext.implicits._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val spark = SparkSession.builder().appName("termAppDF").getOrCreate()

val timeSchema = new StructType(Array(
    new StructField("bigclass",StringType,false),
    new StructField("middle",StringType,false),
    new StructField("total",FloatType,true),
    new StructField("0to3",FloatType,true),
    new StructField("3to6",FloatType,true),
    new StructField("6to9",FloatType,true),
    new StructField("9to12",FloatType,true),
    new StructField("12to15",FloatType,true),
    new StructField("15to18",FloatType,true),
    new StructField("18to21",FloatType,true),
    new StructField("21to24",FloatType,true),
    new StructField("unknown",FloatType,true)
    ))

val timeDF = spark.read.schema(timeSchema).csv("/termdata/crimedata/crime_time.csv")
timeDF.printSchema()
timeDF.count()

timeDF.createOrReplaceTempView("time")
val timeSQL = spark.sql("SELECT * FROM time ")
timeSQL.show
val daySchema = new StructType(Array(
    new StructField("bigclass",StringType,false),
    new StructField("middle",StringType,false),
    new StructField("total",FloatType,true),
    new StructField("sun",FloatType,true),
    new StructField("mon",FloatType,true),
    new StructField("tue",FloatType,true),
    new StructField("wed",FloatType,true),
    new StructField("thu",FloatType,true),
    new StructField("fri",FloatType,true),
    new StructField("sat",FloatType,true)
    ))

val dayDF = spark.read.schema(daySchema).csv("/termdata/crimedata/crime_day.csv")
dayDF.show()
dayDF.createOrReplaceTempView("day")
val daySQL = spark.sql("SELECT * FROM day ")
val join = spark.sql("select day.bigclass, time.middle, day.sun,day.mon,day.tue,day.wed,day.thu,day.fri,day.sat,time.0to3,time.3to6,time.6to9,time.9to12,time.12to15,time.15to18,time.18to21,time.21to24,time.unknown from day join time on day.middle=time.middle where time.middle != 'traffic' and time.unknown > 1000")

join.show()
val bigcrimeSchema = new StructType(Array(
    new StructField("subdivid",StringType,false),
    new StructField("crime_type",StringType,false),
    new StructField("occurrencearrest",StringType,false),
    new StructField("num",FloatType,true)
    ))

val big5DF = spark.read.schema(bigcrimeSchema).csv("/termdata/crimedata/big5crime.csv")
big5DF.show()
big5DF.createOrReplaceTempView("big5")
val big5SQL = spark.sql("SELECT * FROM big5")
val arr = spark.sql("SELECT subdivid, sum(num) FROM big5 where occurrencearrest = 'arrest' group by subdivid order by sum(num) desc")
arr.show(5)
val avg_arrest =  spark.sql("SELECT avg(num) FROM big5 where occurrencearrest='arrest'")
avg_arrest.show()
val occur = spark.sql("SELECT subdivid ,sum(num) FROM big5 where occurrencearrest='occurrence' group by subdivid order by sum(num) desc")
occur.show(5)
val avg_occur =  spark.sql("SELECT avg(num) FROM big5 where occurrencearrest='occurrence'")
avg_occur.show()

val ar_oc = spark.sql("SELECT occurrencearrest, sum(num) FROM big5 group by occurrencearrest order by sum(num) asc")
ar_oc.show()
ar_oc.write.format("csv").save("/termdata/output1")
val arocSchema = new StructType(Array(
    new StructField("occurrencearrest",StringType,false),
    new StructField("num",FloatType,true)
    ))

val arocDF = spark.read.schema(arocSchema).csv("/termdata/output1/*.csv")
arocDF.createOrReplaceTempView("aroc")
arocDF.show()
val arocSQL = spark.sql("SELECT * FROM aroc")
val ar_ag = spark.sql("SELECT sum(num),83056/sum(num) FROM aroc")
ar_ag.show()
val reg = spark.sql("select subdivid,sum(num) from big5 where occurrencearrest = 'arrest'group by subdivid order by sum(num) asc")
reg.show()
val citySchema = new StructType(Array(
    new StructField("subdivid",StringType,false),
    new StructField("num",FloatType,true),
    new StructField("lat",FloatType,true),
    new StructField("lng",FloatType,true)
    ))

val cityDF = spark.read.schema(citySchema).csv("/termdata/crimedata/big5crimecity.csv")
cityDF.createOrReplaceTempView("city")
val citySQL = spark.sql("select * from city where subdivid != 'null'")
citySQL.createOrReplaceTempView("citynon")

val cctvSchema = new StructType(Array(
    new StructField("manager",StringType, false),
    new StructField("road_address",StringType, false),
    new StructField("address",StringType, false),
    new StructField("object",StringType, false),
    new StructField("c_num",IntegerType, false),
    new StructField("fixel",IntegerType, false),
    new StructField("direction",IntegerType, false),
    new StructField("store",IntegerType, false),
    new StructField("installday",StringType, false),
    new StructField("phone",StringType,false),
    new StructField("x",FloatType, false),
    new StructField("y",FloatType, false),
    new StructField("data_day",StringType, false)
    ))
    
val cctvDF = spark.read.schema(cctvSchema).csv("/termdata/cctv/2018cctv.csv")
cctvDF.createOrReplaceTempView("cctv")
val cctvSQL = spark.sql("SELECT manager,road_address,object,c_num,fixel,direction,store,installday,phone,x,y,data_day FROM cctv")
cctvSQL.show

val placeSchema = new StructType(Array(
    new StructField("bigclass",StringType,false),
    new StructField("middle",StringType,false),
    new StructField("total",FloatType,true),
    new StructField("apt",FloatType,true),
    new StructField("single_house",FloatType,true),
    new StructField("highway",FloatType,true),
    new StructField("road",FloatType,true),
    new StructField("department_store",FloatType,true),
    new StructField("supermarket",FloatType,true),
    new StructField("convenience_store",FloatType,true),
    new StructField("large_discount_store",FloatType,true),
    new StructField("store",FloatType,true),
    new StructField("market",FloatType,true),
    new StructField("accommodation",FloatType,true),
    new StructField("nightlife",FloatType,true),
    new StructField("office",FloatType,true),
    new StructField("factory",FloatType,true),
    new StructField("mine",FloatType,true),
    new StructField("warehouse",FloatType,true),
    new StructField("station",FloatType,true),
    new StructField("subway",FloatType,true),
    new StructField("other_table",FloatType,true),
    new StructField("showcase",FloatType,true),
    new StructField("park",FloatType,true),
    new StructField("school",FloatType,true),
    new StructField("financial_institution",FloatType,true),
    new StructField("medical_institution",FloatType,true),
    new StructField("religious institution",FloatType,true),
    new StructField("mountain",FloatType,true),
    new StructField("sea",FloatType,true),
    new StructField("troop",FloatType,true),
    new StructField("detention_place",FloatType,true),
    new StructField("open_space",FloatType,true),
    new StructField("parking_lot",FloatType,true),
    new StructField("public_toilet",FloatType,true),
    new StructField("pc_cafe",FloatType,true),
    new StructField("etc",FloatType,true)
    ))

val placeDF = spark.read.schema(placeSchema).csv("/termdata/crimedata/crime_place.csv")
placeDF.show()
placeDF.createOrReplaceTempView("place")
val placeSQL = spark.sql("SELECT bigclass, middle, department_store, supermarket, convenience_store,large_discount_store,store,market FROM place ")
placeSQL.show
val violent_market = spark.sql("SELECT bigclass, middle , market, department_store FROM place WHERE bigclass = 'violent'")
violent_market.show
val market = spark.sql("SELECT Rank() Over(order by market desc) as Rank, bigclass, middle , market FROM place WHERE bigclass = 'violent' group by bigclass, market, middle")
market.show
val mlexSchema = new StructType(Array(
    new StructField("DOLEVELNAME",StringType,false),
    new StructField("ROAD_NAME",StringType,false),
    new StructField("SPOT",FloatType,true),
    new StructField("STNAME",StringType,true),
    new StructField("ENDNAME",StringType,true),
    new StructField("YEAR",FloatType,true),
    new StructField("MONTH",FloatType,true),
    new StructField("DAY",FloatType,true),
    new StructField("WEEKDAY",FloatType,true),
    new StructField("DIRECTION",FloatType,true),
    new StructField("1H",FloatType,true),
    new StructField("2H",FloatType,true),
    new StructField("3H",FloatType,true),
    new StructField("4H",FloatType,true),
    new StructField("5H",FloatType,true),
    new StructField("6H",FloatType,true),
    new StructField("7H",FloatType,true),
    new StructField("8H",FloatType,true),
    new StructField("9H",FloatType,true),
    new StructField("10H",FloatType,true),
    new StructField("11H",FloatType,true),
    new StructField("12H",FloatType,true),
    new StructField("13H",FloatType,true),
    new StructField("14H",FloatType,true),
    new StructField("15H",FloatType,true),
    new StructField("16H",FloatType,true),
    new StructField("17H",FloatType,true),
    new StructField("18H",FloatType,true),
    new StructField("19H",FloatType,true),
    new StructField("20H",FloatType,true),
    new StructField("21H",FloatType,true),
    new StructField("22H",FloatType,true),
    new StructField("23H",FloatType,true),
    new StructField("24H",FloatType,true),
    new StructField("TOTAL",FloatType,true),
    new StructField("DOLEVEL",FloatType,true),
    new StructField("VOL",StringType,true)
    ))

val mlexDF = spark.read.schema(mlexSchema).csv("/termdata/mlex.csv")
//mlexDF.createOrReplaceTempView("mlex")
//val mlexSQL = spark.sql("select * from mlex where DOLEVELNAME != 'null'")
mlexDF.printSchema()

//mlexDF.createOrReplaceTempView("mlex")
//val mlexSQL = spark.sql("select * from mlex where DOLEVELNAME != 'null'")
mlexDF.printSchema()
mlexDF.show(2)
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
var tmpdf = sqlContext.read.format("com.databricks.spark.csv")
    .option("header","true")
    .option("inferSchema","true")
    .load(trainData)
    .toDF(columns: _* )
    .cache()

val df = mlexDF.drop("DOLEVELNAME").drop("ROAD_NAME").drop("SPOT").drop("STNAME").drop("ENDNAME").drop("YEAR")
df.show(1)
val VOLIndexer = new StringIndexer().setInputCol("VOL").setOutputCol("VOLIndex")
var demonModel = VOLIndexer.fit(df)
demonModel.transform(df).select("VOL","VOLIndex").show(10)
val assembler = new VectorAssembler()
    .setInputCols(Array("1H","2H","3H","4H","5H","6H","7H","8H","9H","10H","11H","12H","13H","14H","15H","16H","17H","18H","19H","20H","21H","22H","23H","24H"))
    .setOutputCol("tmpFeatures")
	val normalizer = new Normalizer().setInputCol("tmpFeatures").setOutputCol("features")
	val logreg = new LogisticRegression().setMaxIter(10)
logreg.setLabelCol("VOLIndex")
val pipeline = new Pipeline().setStages(Array(VOLIndexer,assembler,normalizer,logreg))
val splits= df.randomSplit(Array(0.7,0.3),seed = 20134069L)
val train = splits(0).cache()
val test = splits(1).cache()
val model = pipeline.fit(train)

var result = model.transform(test)
result.show(1)
val predictionAndLabels = result.select(result("prediction"), result("VOLIndex")).as[(Double, Double)].rdd
val metrics = new BinaryClassificationMetrics(predictionAndLabels)

println(s"percent = ${metrics.areaUnderROC()}\n")