
function makeSomeMaps() {
    map = d3.carto.map();

    d3.select("#map").call(map);
    d3.select("#d3MapPanBox > #left").html("&#9664;")
    d3.select("#d3MapPanBox > #right").html("&#9654;")
    d3.select("#d3MapPanBox > #up").html("&#9650;")
    d3.select("#d3MapPanBox > #down").html("&#9660;")

    terrainLayer = d3.carto.layer.tile();
    terrainLayer
    .path("elijahmeeks.map-azn21pbi")
    .label("東京都の地図")

    // csvLayer = d3.carto.layer.csv();
    // csvLayer
    // .path("csvfiles/201501310815.csv")
    // .label("Tweet")
    // .renderMode("canvas")
    // .markerColor("black")
    // .markerSize(.5)
    // .x("lng")
    // .y("lat")
    // // .on("load", makeHexbins)

    map.addCartoLayer(terrainLayer);
    // map.addCartoLayer(csvLayer)
    map.setScale(9)
    map.centerOn([139.567265, 35.71844], "latlong")
    // map.zoomTo([[139.35645, 35.558440],[139.778080, 35.818440]],"latlong",.7, 5000);


    function makeHexbins() {
        var colorScale = d3.scale.linear().domain([1, 5, 30]).range(["green", "yellow", "red"])
        // hexbinLayerSmall = map.createHexbinLayer(csvLayer, 0.01);
        // hexbinLayerLarge = map.createHexbinLayer(csvLayer, 0.004);
        hexbinLayerLarge = map.createHexbinLayer(tweetLayer, 0.05);

        // hexbinLayerSmall
        // .label("S Degree")
        // .visibility(false)
        // .on("load", function() {hexbinLayerSmall.g().selectAll("path").style("opacity", .5).style("fill", function(d) {return colorScale(d.properties.node.length)})})

        hexbinLayerLarge
        .label("L Degree")
        .on("load", function() {hexbinLayerLarge.g().selectAll("path").style("opacity", .5).style("fill", function(d) {return colorScale(d.properties.node.length)})})

        // map.addCartoLayer(hexbinLayerSmall);
        map.addCartoLayer(hexbinLayerLarge);

    }

    var count = 0
    layerArray = []
    // layerArray.push("csvfiles/201501310820.csv");
    // layerArray.push("csvfiles/201501310825.csv");
    // layerArray.push("csvfiles/201501310830.csv");
    // layerArray.push("csvfiles/201501310835.csv");
    for (var i = 1; i <= 31; i++) {
        for (var j = 0;  j < 24; j++) {
            for (var k = 0; k < 12; k++) {
                if(i < 10) {
                    day = "0" + i
                } else {
                    day = i
                }
                if (j < 10) {
                    hour = "0" + j
                } else {
                    hour = j
                }
                if (k < 2) {
                    minute = "0" + k*5
                } else {
                    minute = k*5
                }
                layerArray.push("csvfiles/201501" + day + hour + minute + ".csv")
            }
        }
    }

    console.log(layerArray)
    tweetLayers = []

    function createTweetLayer() {
        tweetLayer = d3.carto.layer.csv();
        tweetLayer
        .path(layerArray[count])
        .label(layerArray[count].substring(13, 21))
        .renderMode("canvas")
        .markerColor("black")
        .markerSize(.5)
        .x("lng")
        .y("lat")
        .on("load", makeHexbins)

        map.addCartoLayer(tweetLayer);
        tweetLayers.push(tweetLayer);
    }
    function deleteTweetLayer() {
        console.log(tweetLayers)
        if(tweetLayers.length > 0) {

            map.deleteCartoLayer(tweetLayers[0]);
            map.deleteCartoLayer(hexbinLayerLarge)
            map.refresh();
            tweetLayers.splice(0, 1);
        console.log(tweetLayers)
        }


    }
    function treatTweetLayer() {
        deleteTweetLayer();
        count += 1
        if(count >= layerArray.length) {
            count = 0
        }
        createTweetLayer();
    }

    function start_interval() {
    repeat = setInterval(treatTweetLayer, 3000);
    }
    function stop_interval() {
        clearInterval(repeat)
    }

    $('#start').click(function() {
        start_interval();
        console.log("start!!")
    })

    $('#stop').click(function() {
        stop_interval();
        console.log("stop!!")
    })
    // setInterval(animation, 5000);


  }


/*
layerArrayを作って，どうにかして５分おき？のレイヤーを生成する．
for i in range(len(layerArray))
    map.addCartoLayer(layerArray[i])
    hoge = map.createHexbinLayer(layerArray[i])
    hoge.label("")
    .on("load", function() {hoge.g().selectAll("path").style("opacity", .5).style("fill", function(d) {return colorScale(d.properties.node.length)})})
    sleep(5)
    map.deleteCartoLayer(layerArray[i])
    map.deleteCartoLayer()
    map.refresh()
setinterval("func();", 1000)
*/
