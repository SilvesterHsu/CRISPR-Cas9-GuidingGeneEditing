(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-common"],{"41cb":function(t,e,a){"use strict";e["a"]={home:[{path:"/home",name:"Home",components:a("bb51")}],results:[{path:"/results/:gene",name:"Results",components:a("b3c3")}],introduction:[{path:"/introduction",name:"Introduction",components:a("e46f")}]}},"4db0":function(t,e,a){},"548c":function(t,e,a){"use strict";var r=function(){var t=this,e=t.$createElement;t._self._c;return t._m(0)},n=[function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{attrs:{id:"app"}},[a("div",{staticClass:"page"},[a("div",{staticClass:"page-title"},[t._v("Introduction")]),a("div",{staticClass:"content-area"},[a("div",{attrs:{id:"h1"}},[t._v("In GGE, we implement this efficiency predicting model via transfer learning for gene knock-out in Danio rerio (danRer11/GRCz11) using CRISPR/Cas9. \n\t\t\tUsers can input the genomic location (usually a not more than 300 bases wide window around a TSS) to find the efficiency prediction and its rank for each guide in this zone. \n\t\t\tThis web application also offers visualization on the results and the results can be download as csv files. \n\t\t\tHave a try and enjoy GGE!\n\t\t\t")]),a("div",{attrs:{id:"h2"}},[t._v("CRISPR/Cas9 scoring")]),a("div",{attrs:{id:"normal-text"}},[a("strong",[t._v("Bad GC: ")]),t._v("<40 or >70"),a("br"),a("strong",[t._v("4nt self-complementary stem found: ")]),t._v("1 for each"),a("br"),a("strong",[t._v("Efficiency: ")]),t._v("efficiency value comes from our prediction algorithm * 100"),a("br"),a("strong",[t._v("CRISPR color thresholds: ")]),t._v("Dark Color >= 59.85 >= Normal Color >= 42.99 >= Light Color"),a("br"),a("br"),t._v("\n\t\t\tMM0 = 0 mismatches, MM1 = 1 mismatch, MM2 = 2 mismatches, MM3 = 3 mismatches\n\t\t\t")])])])])}];a.d(e,"a",function(){return r}),a.d(e,"b",function(){return n})},"561f":function(t,e,a){"use strict";var r=a("4db0"),n=a.n(r);n.a},"59c5":function(t,e,a){"use strict";var r=a("ef7b"),n=a.n(r);n.a},"5f83":function(t,e,a){"use strict";a.r(e);var r=a("f7a4"),n=a.n(r);for(var s in r)"default"!==s&&function(t){a.d(e,t,function(){return r[t]})}(s);e["default"]=n.a},b3c3:function(t,e,a){"use strict";a.r(e);var r=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{attrs:{id:"app"}},[a("div",{ref:"canvas",staticClass:"site-canvas"},[a("div",{staticClass:"dna"},t._l(t.sequence,function(t,e){return a("div",{staticClass:"dna_section"},[a("div",{staticClass:"node top",style:{"animation-delay":-300*e+"ms"}}),a("div",{staticClass:"node bottom",style:{"animation-delay":-300*e+"ms"}})])}),0)]),a("div",{ref:"dna_text",staticClass:"dna_text"},[t._v("Loading...")]),a("div",{directives:[{name:"show",rawName:"v-show",value:t.render,expression:"render"}],ref:"error",staticClass:"error"},[a("div",{staticClass:"error-title"},[t._v("ERROR")]),t._m(0)]),a("div",{staticClass:"page-results"},[a("div",{directives:[{name:"show",rawName:"v-show",value:t.show_results,expression:"show_results"}],staticClass:"area"},[t._v(t._s(t.area))]),a("div",{ref:"panel",staticClass:"panel"}),a("div",{ref:"plotPanel",staticClass:"plot-panel"}),a("div",{ref:"heatMapPanel",staticClass:"heatMap-panel"},[a("div",{ref:"heatMapHigh",staticClass:"heatMap-high"}),a("div",{ref:"heatMapLow",staticClass:"heatMap-low"})]),a("div",{directives:[{name:"show",rawName:"v-show",value:t.show_results,expression:"show_results"}],staticClass:"download"},[a("button",{staticClass:"download-button",on:{click:t.download}},[t._v("Download Results as .csv")]),a("a",{staticClass:"hyperLink",attrs:{href:t.hyperLink}},[t._v("for more details about this page~")])]),a("div",{directives:[{name:"show",rawName:"v-show",value:t.show_results,expression:"show_results"}],staticClass:"table-area"},[a("div",{staticClass:"tableFixed"})])])])},n=[function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"error-content"},[t._v("\n\t\t\tCan not found the results corresponding to the input batch zone!"),a("br"),t._v("\n\t\t\tThe input must be a no more than 300 bases window around a TSS,"),a("br"),t._v("\n\t\t\te.g. chr13:39276927-39277227\n\t\t")])}],s=(a("6d67"),a("55dd"),a("a481"),a("96cf"),a("3b8d")),i=(a("57e7"),a("28a5"),a("bd86")),o=a("5698"),c=a("bc3a"),l=a.n(c),h={data:function(){var t;return t={hyperLink:"",render1:!1,render2:!1,render:!1,show_results:!1,sequence:[],parsedCSV:"",area:"",fileName:"",query:{isIsoform:!1},cutcoords:"",tss:"",genemin:"",genemax:"",w:"",xScale:"",xScale2:"",margin:{top:20,bottom:20,left:20,right:20}},Object(i["a"])(t,"w",""),Object(i["a"])(t,"tierPx",17),Object(i["a"])(t,"genePx",30),Object(i["a"])(t,"tiers",""),Object(i["a"])(t,"h",""),Object(i["a"])(t,"iso_h",""),Object(i["a"])(t,"xAxis",""),Object(i["a"])(t,"xAxisGrid",""),Object(i["a"])(t,"maxScaleFactor",""),Object(i["a"])(t,"sExtM",""),Object(i["a"])(t,"zoom",""),Object(i["a"])(t,"svg",""),Object(i["a"])(t,"gX",""),Object(i["a"])(t,"gXgrid",""),Object(i["a"])(t,"colorscale",""),Object(i["a"])(t,"chartBody",""),Object(i["a"])(t,"batch_coor",""),Object(i["a"])(t,"batch_zone_bar",""),Object(i["a"])(t,"textRank",""),Object(i["a"])(t,"focus",""),Object(i["a"])(t,"tssArrow",""),Object(i["a"])(t,"tssTick",""),Object(i["a"])(t,"asc",!1),Object(i["a"])(t,"lowHighThreshold",52.09),Object(i["a"])(t,"high_freq",[.3,.28,.35,.26,.26,.25,.27,.29,.36,.44,.39,.5,.36,.5,.38,.16,.7,.19,.29,.07,.18,0,0,.22,.23,.29,.2,.32,.19,.21,.28,.22,.18,.15,.18,.13,.25,.24,.06,.07,.1,.15,.02,.1,0,0,.31,.25,.27,.26,.2,.3,.27,.23,.24,.23,.31,.18,.24,.05,.22,.06,.03,.16,.28,.9,.17,1,1,.17,.24,.1,.28,.22,.26,.24,.2,.19,.15,.16,.13,.27,.2,.15,.72,.19,.55,.29,.01,.55,0,0]),Object(i["a"])(t,"low_freq",[.21,.25,.23,.27,.27,.24,.26,.28,.22,.2,.18,.12,.2,.19,.19,.15,.08,.1,.13,.24,.29,0,0,.32,.35,.28,.4,.37,.39,.38,.35,.36,.39,.45,.49,.44,.34,.34,.58,.81,.84,.81,.46,.53,0,0,.15,.22,.17,.15,.17,.17,.17,.19,.18,.22,.17,.18,.19,.26,.18,.15,.06,.04,.04,.04,.1,1,1,.32,.18,.33,.17,.19,.2,.2,.18,.23,.19,.19,.21,.17,.22,.29,.12,.05,.03,.01,.26,.08,0,0]),t},computed:{},filters:{},created:function(){},mounted:function(){this.sequence=new Array(Math.round(document.body.clientWidth/70)),this.drawWholePage()},methods:{drawSeqOnHeatMap:function(t,e){var a=this.$refs.heatMapPanel.offsetWidth/2,r=30,n={top:20,bottom:20,left:(a-24*r)/2,right:(a-24*r)/2},s=["A","T","G","C"];o["m"](e).selectAll(".pointsHeatMap").data(t.split("")).enter().append("circle").attr("cx",function(t,e){return e*r+n.left+r/2}).attr("cy",function(t,e){return s.indexOf(t)*r+80+r/2}).attr("r",r/8).attr("class","heatMap-points").attr("ref","heatPoints").style("fill","rgb(110,99,105)")},drawHeatMap:function(){var t=this.$refs.heatMapPanel.offsetWidth/2,e=30,a={top:20,bottom:20,left:(t-24*e)/2,right:(t-24*e)/2},r=o["m"](".heatMap-high").append("svg:svg").attr("class","svg-high").attr("width",t).attr("height","200px"),n=(r.selectAll(".heatMapTitle").data(["High Efficiency"]).enter().append("text").text(function(t){return t}).attr("x",t/2).attr("y",40).attr("font-size",30).attr("fill","rgb(157,112,114)").style("text-anchor","middle"),r.selectAll(".baseLabel").data(["A","T","G","C"]).enter().append("text").text(function(t){return t}).attr("x",a.left).attr("y",function(t,a){return a*e+80}).style("text-anchor","end").attr("transform","translate(-6,"+e/1.5+")").attr("class",function(t){return"base-"+t}),r.selectAll(".posLabel").data([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]).enter().append("text").text(function(t){return t}).attr("x",function(t,r){return r*e+a.left}).attr("y",75).style("text-anchor","middle").attr("transform","translate("+e/2+", -6)").attr("class",function(t){return"pos_index"+t}),r.selectAll(".freqHigh").data(this.high_freq).enter().append("rect").attr("x",function(t,r){return r%23*e+a.left}).attr("y",function(t,a){return parseInt(a/23)*e+80}).attr("rx",0).attr("ry",0).attr("class","high-rect").attr("width",e).attr("height",e).style("fill",function(t){return"rgba(157,112,114,"+t+")"}),o["m"](".heatMap-low").append("svg:svg").attr("class","svg-low").attr("width",t).attr("height","200px"));n.selectAll(".heatMapTitle").data(["Low Efficiency"]).enter().append("text").text(function(t){return t}).attr("x",t/2).attr("y",40).attr("font-size",30).attr("fill","rgb(44,62,80)").style("text-anchor","middle"),n.selectAll(".baseLabel").data(["A","T","G","C"]).enter().append("text").text(function(t){return t}).attr("x",a.left).attr("y",function(t,a){return a*e+80}).style("text-anchor","end").attr("transform","translate(-6,"+e/1.5+")").attr("class",function(t){return"base-"+t}),n.selectAll(".posLabel").data([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]).enter().append("text").text(function(t){return t}).attr("x",function(t,r){return r*e+a.left}).attr("y",75).style("text-anchor","middle").attr("transform","translate("+e/2+", -6)").attr("class",function(t){return"pos_index"+t}),n.selectAll(".freqLow").data(this.low_freq).enter().append("rect").attr("x",function(t,r){return r%23*e+a.left}).attr("y",function(t,a){return parseInt(a/23)*e+80}).attr("rx",0).attr("ry",0).attr("class","low-rect").attr("width",e).attr("height",e).style("fill",function(t){return"rgba(44,62,80,"+t+")"})},drawPlot:function(){var t=this.$refs.plotPanel.offsetWidth,e={top:20,bottom:30,left:20,right:50},a=this.parsedCSV.length,r=o["m"](".plot-panel").append("svg:svg").attr("width",t+this.margin.left+this.margin.right).attr("height","150px"),n=o["k"]().domain([1,a]).range([0,t-e.left-e.right]),s=o["k"]().domain(o["h"](this.parsedCSV,function(t){return parseInt(t["GC content (%)"])})).range([150-e.top-e.bottom,0]),i=o["b"](n).ticks(a),c=o["c"](s).ticks(5);r.append("g").attr("class","plot-xAxis").attr("transform","translate("+e.left+","+(150-e.top)+")").call(i),r.append("g").attr("class","plot-yAxis").attr("transform","translate("+e.left+","+e.bottom+")").call(c),r.selectAll(".plot-xLabel").data(["Rank"]).enter().append("text").text(function(t){return t}).attr("x",n(a)+10).attr("y",5).attr("font-size",15).attr("transform","translate("+e.left+","+(150-e.top)+")"),r.selectAll(".plot-yLabel").data(["GC content (%)"]).enter().append("text").text(function(t){return t}).attr("x",-e.left).attr("y",30).attr("font-size",15).attr("transform","translate("+e.left+",0)");var l=this.parsedCSV,h=(r.selectAll(".line").data(this.parsedCSV).enter().append("line").attr("style","stroke:rgba(161,185,189,0.75)").attr("curve","curve").attr("transform","translate("+e.left+","+e.bottom+")").attr("x1",function(t,e){return n(e+1)}).attr("y1",function(t,e){return s(l[e]["GC content (%)"])}).attr("x2",function(t,e){return e+1<l.length?n(e+2):n(e+1)}).attr("y2",function(t,e){return e+1<l.length?s(l[e+1]["GC content (%)"]):s(l[e]["GC content (%)"])}),o["i"]().curve(o["e"].alpha(0)),["rgb(161,185,189)","rgb(177,173,126)","rgb(135,144,153)","rgb(186,182,197)","rgb(172,146,169)"]);r.selectAll("circle").data(this.parsedCSV).enter().append("circle").attr("transform","translate("+e.left+","+e.bottom+")").attr("cx",function(t){return n(t.Rank)}).attr("cy",function(t){return s(t["GC content (%)"])}).attr("r",5).attr("fill",function(t,e){return h[e%5]})},sleep:function(t){return new Promise(function(e){return setTimeout(e,t)})},drawWholePage:function(){var t=Object(s["a"])(regeneratorRuntime.mark(function t(){var e,a,r,n,s,i,c;return regeneratorRuntime.wrap(function(t){while(1)switch(t.prev=t.next){case 0:if(!this.$route.query.gene){t.next=12;break}return this.area=this.$route.query.gene,this.fileName=this.area.replace(":","_"),e=window.location.href.split("?")[0].replace("results","introduction"),this.hyperLink=e,t.next=7,this.loadCSV();case 7:return t.next=9,this.getCutcoords();case 9:this.render1&&this.render2?(this.$refs.canvas.remove(),this.$refs.dna_text.remove(),this.show_results=!0,a=o["m"](".tableFixed").append("table").attr("class","wholeTable"),r=a.append("thead"),n=a.append("tbody"),r.append("tr").selectAll("th").data(this.parsedCSV.columns).enter().append("th").text(function(t){return t}),i=this,s=n.selectAll("tr").data(this.parsedCSV).enter().append("tr").attr("id",function(t){return"guide"+t["Rank"]}).on("mouseover",function(t){o["m"]("#guide-line"+t["Rank"]).attr("stroke-width",15),o["m"]("body").style("cursor","pointer"),o["m"]("#guide"+t["Rank"]).selectAll("td").style("background-color","rgba(186,182,197,0.3)"),i.focus.style("display",null).select(".zoombox").attr("x1",i.xScale(i.cutcoords[parseInt(t["Rank"])-1][1]-1)).attr("x2",i.xScale(i.cutcoords[parseInt(t["Rank"])-1][1]+i.cutcoords[parseInt(t["Rank"])-1][3]-1)),i.drawSeqOnHeatMap(t["Target sequence"],t["Efficiency"]>=i.lowHighThreshold?".svg-high":".svg-low")}).on("mouseout",function(t){o["m"]("body").style("cursor","move"),o["m"]("#guide-line"+t["Rank"]).attr("stroke-width",10),o["m"]("#guide"+t["Rank"]).selectAll("td").style("background-color","white");for(var e=document.getElementsByClassName("heatMap-points"),a=e.length-1;a>=0;a--)e[a].parentNode.removeChild(e[a])}),i=this,r.selectAll("th").on("click",function(t){var e=i.asc?o["a"]:o["f"],a=i.asc?-1:1;i.asc=!i.asc,s.sort(function(r,n){var s=null;if(/\d/.test(r[t])){var i=r[t].replace(/\D/g,""),o=n[t].replace(/\D/g,"");s=a*(i-o)}else s=e(n[t],r[t]);return s})}),s.selectAll("td").data(function(t){return i.parsedCSV.columns.map(function(e){var a=t[e];return{column:e,value:a}})}).enter().append("td").html(function(t){return"<code>"+t.value+"</code>"}),this.w=this.$refs.panel.offsetWidth-this.margin.left-this.margin.right,this.tiers=o["j"](this.cutcoords,function(t){return t[t.length-1]}),this.h=Math.max(this.tiers,5)*this.tierPx+10+30-this.margin.top-this.margin.bottom,this.iso_h=Math.max(this.tiers,5)*this.tierPx+10-this.margin.top,this.xScale=o["k"]().domain([this.genemin,this.genemax]).range([0,this.w]),this.xScale2=o["k"]().domain([this.genemin,this.genemax]).range([0,this.w]),this.xAxis=o["b"](this.xScale).ticks(10),this.xAxisGrid=o["b"](this.xScale).ticks(10).tickSize(-this.h,0,0).tickFormat(""),this.maxScaleFactor=this.w/Math.abs(this.genemin-this.genemax),this.sExtM=this.w/Math.abs(this.xScale(100)-this.xScale(1)),this.zoom=o["n"]().scaleExtent([1,this.sExtM]).on("zoom",this.zoomed),this.svg=this.createPicZone(),this.gX=this.drawNumLabelBelow(),this.gXgrid=this.createGrid(),this.clip=this.createClip(),this.focus=this.drawFocusShadow(),this.chartBody=this.svg.append("g").attr("clip-path","url(#clip)"),this.createMarker(),this.colorscale=o["l"]().domain([42.97,59.85]).range(["rgba(44,62,80,1)","rgba(44,62,80,0.75)","rgba(44,62,80,0.5)"]),this.drawMainBarchart(),this.batch_zone_bar=this.drawBatchZone(),this.textRank=this.drawRankLabel(),this.drawLegend(),c=this.svg.append("g").attr("class","tssArea"),this.tssArrow=this.drawTssArrow(c),this.tssTick=this.drawTssTick(c),this.$refs.panel.style.borderStyle="dashed",this.$refs.panel.style.borderColor="#2c3e50",this.$refs.panel.style.borderWidth="1px",this.drawPlot(),this.drawHeatMap()):(this.$refs.canvas.remove(),this.$refs.dna_text.remove()),t.next=15;break;case 12:this.$refs.canvas.remove(),this.$refs.dna_text.remove(),this.render=!0;case 15:case"end":return t.stop()}},t,this)}));function e(){return t.apply(this,arguments)}return e}(),getCutcoords:function(){var t=Object(s["a"])(regeneratorRuntime.mark(function t(){var e,a=this;return regeneratorRuntime.wrap(function(t){while(1)switch(t.prev=t.next){case 0:return e=window.location.href.split("?")[0].replace("results","picture"),t.next=3,l.a.post(e+"/"+this.fileName).then(function(t){200===t.status?""===t.data.genemax?(a.render1=!1,a.render=!0):(a.render1=!0,a.cutcoords=t.data.cutcoords,a.tss=t.data.tss,a.genemin=t.data.genemin,a.genemax=t.data.genemax,a.batch_coor=t.data.batch_coor):a.render=!0});case 3:case"end":return t.stop()}},t,this)}));function e(){return t.apply(this,arguments)}return e}(),loadCSV:function(){var t=Object(s["a"])(regeneratorRuntime.mark(function t(){var e;return regeneratorRuntime.wrap(function(t){while(1)switch(t.prev=t.next){case 0:return e=window.location.href.split("?")[0],t.prev=1,t.next=4,o["d"](e+"/"+this.fileName+".csv");case 4:this.parsedCSV=t.sent,t.next=12;break;case 7:t.prev=7,t.t0=t["catch"](1),this.render=!0,this.$refs.canvas.remove(),this.$refs.dna_text.remove();case 12:this.parsedCSV.length>0?this.render2=!0:this.render=!0;case 13:case"end":return t.stop()}},t,this,[[1,7]])}));function e(){return t.apply(this,arguments)}return e}(),zoomed:function(){var t=this,e=this;o["g"].sourceEvent&&"brush"===o["g"].sourceEvent.type||(this.xScale.domain(o["g"].transform.rescaleX(this.xScale2).domain()),this.gX.call(this.xAxis),this.gXgrid.call(this.xAxisGrid),this.svg.selectAll(".cutsite").attr("x1",function(e){return t.xScale(e[1]-1)}).attr("x2",function(e){return t.xScale(e[1]+e[3]-1)}),this.batch_zone_bar.attr("x1",function(e){return t.xScale(e[1])}).attr("x2",function(e){return t.xScale(e[2])}),this.textRank.attr("x",function(e){return t.xScale(e[1]+e[3]/2)}),this.tssArrow.attr("points",function(t){var a=[e.xScale(t)+.5,",",e.h-5," ",e.xScale(t)-2.5,",",e.h+5," ",e.xScale(t)+3.5,",",e.h+5];return a.join("")}),this.tssTick.attr("x1",function(e){return t.xScale(e)+.5}).attr("x2",function(e){return t.xScale(e)+.5}).attr("y1","0").attr("y2",this.h))},createPicZone:function(){var t=o["m"](".panel").append("svg:svg").attr("width",this.w+this.margin.left+this.margin.right).attr("height",this.h+this.margin.top+this.margin.bottom).call(this.zoom).append("svg:g").attr("transform","translate("+this.margin.left+","+this.margin.top+")");return t},drawNumLabelBelow:function(){var t=this.svg.append("svg:g").attr("class","x axis").attr("transform","translate(0, "+this.h+")").call(this.xAxis);return t},createGrid:function(){var t=this.svg.append("g").attr("class","x grid").attr("transform","translate(0,"+this.h+")").call(this.xAxisGrid);return t},createClip:function(){var t=this.svg.append("svg:clipPath").attr("id","clip").append("svg:rect").attr("x",0).attr("y",0).attr("width",this.w).attr("height",this.h);return t},createMarker:function(){this.svg.append("defs").append("marker").data(this.cutcoords).attr("id","forward_arrow").attr("refX",0).attr("refY",.5).attr("markerWidth",6).attr("markerHeight",1).attr("stroke-width",0).attr("orient","auto").append("path").attr("d","M 0,0 V 1 L0.5,0.5 Z").style("stroke","rgba(61,49,91,1)").style("fill","rgba(61,49,91,1)"),this.svg.append("defs").append("marker").data(this.cutcoords).attr("id","reverse_arrow").attr("refX",.5).attr("refY",.5).attr("markerWidth",6).attr("markerHeight",2).attr("stroke-width",0).attr("orient","auto").append("path").attr("d","M 0.5,0 V 1 L0,0.5 Z").style("stroke","rgba(61,49,91,1)").style("fill","rgba(61,49,91,1)")},drawMainBarchart:function(){var t=this,e=this;this.chartBody.selectAll("cutsite").data(this.cutcoords).enter().append("line").attr("marker-start",function(e){var a=!t.query.isIsoform&"-"==e[4];return a?"url(#reverse_arrow)":"none"}).attr("marker-end",function(e){var a=!t.query.isIsoform&"+"==e[4];return a?"url(#forward_arrow)":"none"}).attr("class","cutsite").attr("id",function(t){return"guide-line"+t[0]}).attr("x1",function(e){return t.xScale(e[1]-1)}).attr("x2",function(e){return t.xScale(e[1]+e[3]-1)}).attr("y1",function(e){return t.iso_h-25-12*e[e.length-1]}).attr("y2",function(e){return t.iso_h-25-12*e[e.length-1]}).attr("stroke-width",10).attr("height",10).attr("stroke",function(e){return t.colorscale(e[2])}).on("mouseover",function(t){o["m"](this).attr("stroke-width",15),o["m"]("body").style("cursor","pointer"),e.focus.style("display",null).select(".zoombox").attr("x1",e.xScale(t[1]-1)).attr("x2",e.xScale(t[1]+t[3]-1)),o["m"]("#guide"+t[0]).selectAll("td").style("background-color","rgba(186,182,197,0.3)")}).on("mouseout",function(t){o["m"]("body").style("cursor","move"),o["m"](this).attr("stroke-width",10),e.focus.style("display","none"),o["m"]("#guide"+t[0]).selectAll("td").style("background-color","white")})},drawBatchZone:function(){var t=this,e=this.chartBody.selectAll("barBlow").data(this.batch_coor).enter().append("g"),a=e.selectAll("zone").data(function(t,e){return t.zone.map(function(t){return t.push(e)}),t.zone}).enter().append("line").attr("x1",function(e){return t.xScale(e[1])}).attr("x2",function(e){return t.xScale(e[2])}).attr("y1",function(e){return t.iso_h}).attr("y2",function(e){return t.iso_h}).attr("stroke","rgba(110,99,105,1)").attr("stroke-width",8);return a},drawRankLabel:function(){var t=this,e=this.chartBody.selectAll("cutsitetext").data(this.cutcoords).enter().append("text").attr("class","cutsitetext").text(function(t){return t[0]}).attr("font-size","10px").attr("stroke","rgba(240,228,206,1)").attr("x",function(e){return t.xScale(e[1]-1+e[3]/2)}).attr("y",function(e){return t.iso_h-10-12-12*e[e.length-1]});return e},drawLegend:function(){var t=this.chartBody.append("g");t.append("rect").attr("x",0).attr("y",0).attr("width",100).attr("height",55).attr("stroke","gray").attr("stroke-width",.5).attr("fill","white").style("z-index",1e5),t.append("text").text("5' --\x3e  3'").attr("x",32).attr("y",13).attr("font-family","arial").attr("font-size","12px").attr("fill","rgba(57,45,99,1)").attr("z-index",1e5),t.append("line").attr("x1",7).attr("x2",37).attr("y1",22).attr("y2",22).attr("stroke-width",8).attr("stroke","rgba(110,99,105,1)").attr("z-index",1e5),t.append("text").text("Batch Zone").attr("x",42).attr("y",25).attr("font-family","arial").attr("font-size","10px").attr("fill","black").attr("z-index",1e5),t.append("line").attr("x1",7).attr("x2",17).attr("y1",35).attr("y2",35).attr("stroke-width",10).attr("stroke","rgba(44,62,80,1)").attr("z-index",1e5),t.append("line").attr("x1",17).attr("x2",27).attr("y1",35).attr("y2",35).attr("stroke-width",10).attr("stroke","rgba(44,62,80,0.75)").attr("z-index",1e5),t.append("line").attr("x1",27).attr("x2",37).attr("y1",35).attr("y2",35).attr("stroke-width",10).attr("stroke","rgba(44,62,80,0.5)").attr("z-index",1e5),t.append("text").text("Target").attr("x",42).attr("y",38).attr("font-family","arial").attr("font-size","10px").attr("fill","black").attr("z-index",1e5),t.append("polygon").attr("class","tssArrow").attr("fill","rgba(148,60,58,1)").attr("points","21,42 18,50 24,50"),t.append("text").text("TSS").attr("x",42).attr("y",50).attr("font-family","arial").attr("font-size","10px").attr("fill","black").attr("z-index",1e5),t.append("text").text("Double click to zoom out.").attr("x",this.w-120).attr("y",10).attr("font-family","arial").attr("font-size","10px").attr("fill","black").attr("z-index",1e5),t.append("text").text("Drag to move left and right.").attr("x",this.w-120).attr("y",20).attr("font-family","arial").attr("font-size","10px").attr("fill","black").attr("z-index",1e5),t.append("text").text(this.batch_coor[0].chromo).attr("stroke","#FFFFFF").attr("stroke-width",.5).attr("x",0).attr("y",this.h-5).attr("font-family","arial").attr("font-size","15px").attr("font-weight","bold").attr("fill","rgba(146,80,82,1)").attr("z-index",1e5)},drawFocusShadow:function(){var t=this.svg.append("g").attr("class","focus").style("display","none");return t.append("line").attr("class","zoombox").attr("x1",100).attr("x2",100).attr("y1",this.iso_h-13).attr("y2",this.iso_h-13).attr("stroke","DarkGray").attr("stroke-width",10).attr("stroke-opacity",.3),t},drawTssArrow:function(t){var e=this,a=t.selectAll("tss").data(this.tss).enter().append("polygon").attr("class","tssArrow").attr("points",function(t){var a=[e.xScale(t)+.5,",",e.h-5," ",e.xScale(t)-2.5,",",e.h+5," ",e.xScale(t)+3.5,",",e.h+5];return a.join("")}).attr("fill","rgba(148,60,58,1)");return a},drawTssTick:function(t){var e=this,a=t.selectAll("tss").data(this.tss).enter().append("line").attr("class","tssGrid").attr("x1",function(t){return e.xScale(t)+.5}).attr("x2",function(t){return e.xScale(t)+.5}).attr("y1","0").attr("y2",this.h).attr("stroke","rgba(148,60,58,1)").attr("stroke-width",1).attr("stroke-dasharray","3 2");return a},download:function(){var t=window.location.href.split("?")[0];window.open(t+"/"+this.fileName+".csv","_blank")}}},d=h,u=(a("59c5"),a("2877")),f=Object(u["a"])(d,r,n,!1,null,null,null);e["default"]=f.exports},bb51:function(t,e,a){"use strict";a.r(e);var r=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{attrs:{id:"app"}},[a("div",{staticClass:"page-home"},[a("div",{staticClass:"logo"}),t._m(0),a("div",{staticClass:"input-area"},[a("div",{staticClass:"input-label"},[t._v("Target:")]),a("div",{staticClass:"input-box"},[a("input",{directives:[{name:"model",rawName:"v-model",value:t.gene,expression:"gene"}],staticClass:"gene",attrs:{placeholder:"Genomic coordinates, e.g. chr11:37719830-37720130"},domProps:{value:t.gene},on:{input:function(e){e.target.composing||(t.gene=e.target.value)}}}),a("button",{staticClass:"clear-input",on:{click:t.clear}},[t._v("reset")])])]),t._m(1),a("div",{staticClass:"submit-area"},[a("button",{staticClass:"submit-button",on:{click:t.submit}},[t._v("Find Target Sites!")])]),a("div",{directives:[{name:"show",rawName:"v-show",value:t.isShow,expression:"isShow"}],staticClass:"alert"},[t._v("Gene does not exist. "),a("br"),t._v("\n\t\tPlease input genomic coordinates."),a("br"),t._v("\n\t\tIt should be a no more than 300 bases window around a TSS,"),a("br"),t._v("\n\t\te.g. chr13:39276927-39277227\n\t\t")])])])},n=[function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"title"},[t._v("GGE "),a("strong",[t._v("GGE")])])},function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"fixed-area"},[a("div",{staticClass:"species"},[t._v("In:"),a("br"),t._v("Danio rerio (danRer11/GRCz11)")]),a("div",{staticClass:"method"},[t._v("Using:"),a("br"),t._v("CRISPR/Cas9")]),a("div",{staticClass:"knock"},[t._v("For:"),a("br"),t._v("knock-out")])])}],s=(a("28a5"),a("a481"),a("96cf"),a("3b8d")),i=a("bc3a"),o=a.n(i),c={name:"home",components:{},data:function(){return{gene:"",isShow:!1,pass:!1,arr:["a","b","c"]}},methods:{submit:function(){var t=Object(s["a"])(regeneratorRuntime.mark(function t(){var e,a,r=this;return regeneratorRuntime.wrap(function(t){while(1)switch(t.prev=t.next){case 0:if(e=/^chr\d{1,2}:\d{5,20}-\d{5,20}$/,!e.test(this.gene)){t.next=8;break}return this.isShow=!1,a=window.location.href.split("?")[0].replace("home","test"),t.next=6,o.a.post(a+"/"+this.gene.replace(":","_")).then(function(t){if(200===t.status)if("666"===t.data.msg){var e=r.$router.resolve({path:"/results",query:{gene:r.gene}});window.open(e.href,"_blank")}else r.isShow=!0;else r.isShow=!0});case 6:t.next=9;break;case 8:this.isShow=!0;case 9:case"end":return t.stop()}},t,this)}));function e(){return t.apply(this,arguments)}return e}(),clear:function(){this.gene=""}}},l=c,h=(a("cccb"),a("2877")),d=Object(h["a"])(l,r,n,!1,null,null,null);e["default"]=d.exports},c0d6:function(t,e){},cccb:function(t,e,a){"use strict";var r=a("d563"),n=a.n(r);n.a},d563:function(t,e,a){},e46f:function(t,e,a){"use strict";a.r(e);var r=a("548c"),n=a("5f83");for(var s in n)"default"!==s&&function(t){a.d(e,t,function(){return n[t]})}(s);a("561f");var i=a("2877"),o=Object(i["a"])(n["default"],r["a"],r["b"],!1,null,null,null);e["default"]=o.exports},ef7b:function(t,e,a){},f7a4:function(t,e){}}]);
//# sourceMappingURL=chunk-common.54919c6c.js.map