var w = 1000;
var h = 1000;

var svg = d3.select('body')
            .append('svg')
            .attr('width', w)
            .attr('height', h);


d3.json('../data/modify_data/tokyo_honshu.topojson', function(json) {
    var japan = topojson.object(json, json.objects.tokyo_honshu).geometries;
    var projection = d3.geo.mercator()
                    .center([139.532454, 35.723002])
                    .translate([w/2-150, h/2-350])
                    .scale(30000);

    var path = d3.geo.path().projection(projection);
    var color = d3.scale.category10();
    // color = d3.scale.linear().domain([0, 38]).range(["#0000FF", "#FFFFFF"]);

    svg.selectAll("path")
        .data(japan)
        .enter()
        .append("path")
        .attr("d", path)
        .attr("stroke", "black")
        .attr("stroke-width", 0.5)
        .attr('class', function(d) {
            return 'area areatokyo';
        })
        // .style("fill", function(d, i) {
        //     // return color(i);
        //     return "#ddd";
        //  })
        .attr('data-cityname', function(d) {
            // console.log(d.properties.name)
            if (d.properties.ku) {
                return d.properties.ku
            }
            return d.properties.shi
        })
        .on('click', function() {
            var self = d3.select(this);
            d3.select('#cityname').text("場所： " + self.attr('data-cityname'))
        })

    svg.selectAll(".place-label")
        .data(japan)
        .enter()
        .append("text")
        .attr("font-size", "8px")
        .attr("class", "place-label")
        .attr("transform", function(d) {
            return "translate(" + path.centroid(d) + ")";
        })
        .attr("dx", "-1.5em")
        .text(function(d) {
            if (d.properties.ku) {
                return d.properties.ku;
            }
            return d.properties.shi;
        });
});
