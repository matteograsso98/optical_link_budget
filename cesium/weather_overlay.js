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
    new Cesium.SingleTileImageryProvider({ url: 'maps/temperature_map.png',    rectangle: RECT })
);
const wind10Layer = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'maps/wind_speed_10m_map.png', rectangle: RECT })
);
const wind50Layer = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'maps/wind_speed_50m_map.png', rectangle: RECT })
);
const wind100Layer = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'maps/wind_speed_100m_map.png', rectangle: RECT })
);
const lwLayer = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'maps/lw_map.png', rectangle: RECT })
);
const cn2Layer = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'maps/cn2_map.png', rectangle: RECT })
);
const attenuationLayer = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({ url: 'maps/attenuation_map.png', rectangle: RECT })
);

const allLayers = { temperature: tempLayer, wind10: wind10Layer, wind50: wind50Layer, wind100: wind100Layer, lw: lwLayer, cn2: cn2Layer, attenuation: attenuationLayer };

// Default Displayed Temperature
Object.values(allLayers).forEach(l => l.show = false);
tempLayer.show = true;

window.showLayer = function(name) {
    Object.entries(allLayers).forEach(([key, layer]) => layer.show = (key === name));
    document.querySelectorAll('#layer-controls button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.layer === name);
    });
};
