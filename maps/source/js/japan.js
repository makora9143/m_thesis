var w = 1000;
var h = 1000;

var svg = d3.select('body')
            .append('svg')
            .attr('width', w)
            .attr('height', h);

d3.json('data/modify_data/japan.topojson', function(json) {
    var japan = topojson.object(json, json.objects.japan).geometries;
    var projection = d3.geo.mercator()
                    .center([137, 34])
                    .translate([w/2, h/2])
                    .scale(1500);
    var color = d3.scale.category10();

    var path = d3.geo.path().projection(projection);

    svg.selectAll("path")
        .data(japan)
        .enter()
        .append("path")
        .attr("d", path)
        .attr('stroke', "black")
        .attr('stroke-width', 0.5)
        .style('fill', function(d, i) {
            return color(i);
        });
    svg.selectAll(".place-label")
        .data(japan)
        .enter()
        .append("text")
        .attr("font-size", "8px")
        .attr("class", "place-label")
        .attr("transform", function(d) {
            var lat = d.properties.latitude;
            var lng = d.properties.longitude;
            return "translate(" + projection([lng, lat]) + ")";
        })
        .attr("dx", "-1.5em")
        .text(function(d) {
            return d.properties.name_local;
        });
});
