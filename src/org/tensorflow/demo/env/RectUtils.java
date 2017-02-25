package org.tensorflow.demo.env;

import android.graphics.Rect;
import android.graphics.RectF;

public class RectUtils {

    public static Rect toRect(RectF rect) {
        return new Rect((int) rect.left, (int) rect.top, (int) rect.right, (int) rect.bottom);
    }

    public static Rect fixRect(int width, int height, RectF rect,
                               float min_height_ratio, float min_width_ratio, float expand_ratio) {
        int left = Math.max((int)(rect.left * width - width * expand_ratio / 2), 0);
        int right = Math.min((int)(rect.right * width + width * expand_ratio / 2), width);
        int top = Math.max((int)(rect.top * height - height * expand_ratio / 2), 0);
        int bottom = Math.min((int)(rect.bottom * height + height * expand_ratio / 2), height);
        if (right - left < width * min_width_ratio || left < 0 || left > width || right < 0 || right > width) {
            left = 0;
            right = width;
        }
        if (bottom - top < height * min_height_ratio || top < 0 || top > height || bottom < 0 || bottom > height) {
            top = 0;
            bottom = height;
        }
        return new Rect(left, top, right, bottom);
    }

    public static Rect fixRect(int width, int height, RectF rect) {
        return fixRect(width, height, rect, 0.2f, 0.2f, 0.075f);
    }
}
