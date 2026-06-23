const viewer = new Cesium.Viewer('cesiumContainer', {
    baseLayerPicker: false,
    imageryProvider: new Cesium.UrlTemplateImageryProvider({
        url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        subdomains: ['a', 'b', 'c']
    }),
    animation: false,
    timeline: false
});

const RECT = Cesium.Rectangle.fromDegrees(-180.0, -90.0, 180.0, 90.0);

const tempLayer = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'temperature_map.png', rectangle: RECT })
);

const windLayer = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'wind_speed_map.png', rectangle: RECT })
);

// Default Displayed Temperature
tempLayer.show = true;
windLayer.show = false;

window.showLayer = function(name) {
    tempLayer.show = (name === 'temperature');
    windLayer.show = (name === 'wind');

    document.getElementById('btn-temp').classList.toggle('active', name === 'temperature');
    document.getElementById('btn-wind').classList.toggle('active', name === 'wind');
};
