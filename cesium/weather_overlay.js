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

const tempLayer   = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'temperature_map.png',    rectangle: RECT })
);
const wind10Layer = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'wind_speed_10m_map.png', rectangle: RECT })
);
const wind50Layer = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'wind_speed_50m_map.png', rectangle: RECT })
);
const wind100Layer = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'wind_speed_100m_map.png', rectangle: RECT })
);

const allLayers = { temperature: tempLayer, wind10: wind10Layer, wind50: wind50Layer, wind100: wind100Layer };

// Température visible par défaut
Object.values(allLayers).forEach(l => l.show = false);
tempLayer.show = true;

window.showLayer = function(name) {
    Object.entries(allLayers).forEach(([key, layer]) => layer.show = (key === name));
    document.querySelectorAll('#layer-controls button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.layer === name);
    });
};
