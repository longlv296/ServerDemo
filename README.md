# Paint by Number data

Branch `PaintByNumber` hosts the artwork data loaded by the Paint by Number app
(`com.asianmobile.paintbynumber`) from
`https://raw.githubusercontent.com/longlv296/ServerDemo/PaintByNumber/`.

```text
catalog.json                          categories + artwork entries (relative URLs)
artworks/<category>/<artwork_id>/
    image.jpg                         source pixels revealed when a region is filled
    thumbnail.jpg                     home screen preview
    line_art.png                      transparent outline overlay
    region_map.png                    region id per pixel (R + G<<8 + B<<16)
    paint_data.json                   palette, regions, labels
```

Content is generated and published from the app repository with
`CaptureDemo/tools/paint_data_tool.py` and
`CaptureDemo/tools/publish_server_data.py`. Do not edit region maps by hand:
when a region map changes, the publisher bumps the artwork's `contentVersion`
so saved progress for the old map is ignored.

Licenses: every non-anime artwork is CC0 or public domain; each catalog entry
records `license`, `creator`, and `sourceUrl`.
