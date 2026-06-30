// ── Satellite base layer — Esri World Imagery (no file, no token) ─────────────
const _baseLayer = Cesium.ImageryLayer.fromProviderAsync(
    Cesium.ArcGisMapServerImageryProvider.fromUrl(
        'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
    )
);

const viewer = new Cesium.Viewer('cesiumContainer', {
    baseLayerPicker: false,
    animation:       false,
    timeline:        false,
    baseLayer:       _baseLayer,
});

// ── GFS overlay layers ────────────────────────────────────────────────────────
const RECT = Cesium.Rectangle.fromDegrees(-180.0, -90.0, 180.0, 90.0);

const allLayers = {
    temperature: viewer.imageryLayers.addImageryProvider(
        new Cesium.SingleTileImageryProvider({ url: 'maps/temperature_map.png',     rectangle: RECT })
    ),
    wind10:      viewer.imageryLayers.addImageryProvider(
        new Cesium.SingleTileImageryProvider({ url: 'maps/wind_speed_10m_map.png',  rectangle: RECT })
    ),
    wind50:      viewer.imageryLayers.addImageryProvider(
        new Cesium.SingleTileImageryProvider({ url: 'maps/wind_speed_50m_map.png',  rectangle: RECT })
    ),
    wind100:     viewer.imageryLayers.addImageryProvider(
        new Cesium.SingleTileImageryProvider({ url: 'maps/wind_speed_100m_map.png', rectangle: RECT })
    ),
    attenuation: viewer.imageryLayers.addImageryProvider(
        new Cesium.SingleTileImageryProvider({ url: 'maps/attenuation_map.png',     rectangle: RECT })
    ),
};

// Start with bare Earth — no overlay active
Object.values(allLayers).forEach(l => l.show = false);

// ── Toggle: click active button → hide; click inactive → show ─────────────────
window.showLayer = function(name) {
    const layer    = allLayers[name];
    const wasActive = layer.show;

    Object.values(allLayers).forEach(l => l.show = false);
    document.querySelectorAll('#layer-controls button').forEach(b => b.classList.remove('active'));

    if (!wasActive) {
        layer.show = true;
        document.querySelector('[data-layer="' + name + '"]').classList.add('active');
    }
};
