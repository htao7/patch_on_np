import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from math import atan2

IMG_R = 3
NP_RADIUS = 20


def nothing(x):
    pass


def calc_mag():  # calculate the magnification
    sb_list = [100, 200, 500, 1000, 2000]
    img_scalebar = []
    img_scalebar.append(cv2.imread("./mag/100nm.png", 0))
    img_scalebar.append(cv2.imread("./mag/200nm.png", 0))
    img_scalebar.append(cv2.imread("./mag/500nm.png", 0))
    img_scalebar.append(cv2.imread("./mag/1um.png", 0))
    img_scalebar.append(cv2.imread("./mag/2um.png", 0))

    img_mag = img[2500:2600, 3100:3296]  # mag num image
    # cv2.imwrite('2um.png',img_mag)
    score_max = 0
    magi = 0
    for i, sb in enumerate(sb_list):
        res = cv2.matchTemplate(img_mag, img_scalebar[i], cv2.TM_CCOEFF)
        (_, score, _, _) = cv2.minMaxLoc(res)
        if score >= score_max:
            score_max = score
            magi = sb
    img_sb = img[2475:2485, 1600:3296]  # scale bar image
    ret, img_sb = cv2.threshold(img_sb, 50, 255, cv2.THRESH_BINARY_INV)
    _, sbcont, _ = cv2.findContours(img_sb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mag = float(magi) / (sbcont[0][0][0][0] - sbcont[10][0][0][0])
    return mag


def ExportData():
    file = open('result.txt', 'a')
    for i in range(len(patch_size)):
        file.write("%s\t%.1f\t%i\t%.1f\t%.1f\n" % (
        imgfile, patch_size[i], patch_aff[i], patch_ang[i][0] % 360, patch_ang[i][1] % 360))
    file.close()


def CalcAngle(p1, p2, p3):
    angle = atan2(-p3[1] + p1[1], p3[0] - p1[0]) - atan2(-p2[1] + p1[1], p2[0] - p1[0])
    return (angle)


# plt.hist(img_blur.ravel(),256,(0,255))
# plt.show()

def ExportInit():
    file = open('result.txt', 'w')
    file.write("Img\tArea\tAttachTo\tAngle1\tAngle2\n")
    file.close()


ExportInit()
if os.path.isdir('./results') is False:
    os.mkdir('./results')
for imgfile in os.listdir('.'):
    if imgfile.endswith('.jpg') or imgfile.endswith('.bmp'):

        cv2.namedWindow('dst')
        cv2.createTrackbar('Thresh_low', 'dst', 30, 60, nothing)
        cv2.createTrackbar('Thresh_high', 'dst', 20, 70, nothing)

        img0 = cv2.imread(imgfile)
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        mag = calc_mag()
        if mag < 0.5:
            iter = 3
        else:
            iter = 2
        r, c = img.shape  # 2788 * 3296
        r2 = int(r / IMG_R)
        c2 = int(c / IMG_R)
        #img = cv2.resize(img, (c2, r2), interpolation=cv2.INTER_AREA)
        img_blur = cv2.medianBlur(img, 25)
        img_blur = cv2.GaussianBlur(img_blur, (25, 25), 0)
        th_flag = tl_flag = -1
        show_flag = 0
        x1 = y1 = 0
        x2 = y2 = 0
        img_resize = cv2.resize(img0, (c2, r2), interpolation=cv2.INTER_AREA)
        canvas = np.zeros(img_blur.shape, np.uint8)
        while (True):

            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                break
            elif k == 32:
                show_flag = (show_flag + 1) % 3

            thresh_low = cv2.getTrackbarPos('Thresh_low', 'dst')
            thresh_high = cv2.getTrackbarPos('Thresh_high', 'dst')

            if thresh_low != tl_flag or thresh_high != th_flag:

                # otsu,img_np = cv2.threshold(img_blur,thresh_low,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                _, blackpart = cv2.threshold(img_blur, thresh_low + 60, 255, cv2.THRESH_BINARY_INV)
                _, cont, _ = cv2.findContours(blackpart, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                np_center_list = []
                np_core = np.copy(canvas)

                for conti in cont:
                    area = cv2.contourArea(conti)
                    (x, y), r = cv2.minEnclosingCircle(conti)
                    if r * mag  > NP_RADIUS * 0.6 and r * mag  < NP_RADIUS * 1.5 and area / (
                            np.pi * r ** 2) > 0.7:
                        cv2.drawContours(np_core, [conti], 0, 255, -1)

                _, core_cont, _ = cv2.findContours(np_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # distance_map = []
                np_core_list = []
                for conti in core_cont:
                    area = cv2.contourArea(conti)
                    (x, y), r = cv2.minEnclosingCircle(conti)
                    x = int(x)
                    y = int(y)
                    r = int(r)
                    np_center_list.append((x, y))
                    npi = np.copy(canvas)
                    cv2.drawContours(npi, [conti], 0, 255, -1)
                    # distance_map.append(cv2.distanceTransform(cv2.bitwise_not(npi), cv2.DIST_L2, 3))
                    np_core_list.append(npi)

                _, graypart = cv2.threshold(img_blur, thresh_high + thresh_low + 80, 255, cv2.THRESH_BINARY_INV)
                _, cont, _ = cv2.findContours(graypart, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                np_patch = np.copy(canvas)
                for conti in cont:
                    for center in np_center_list:
                        if cv2.pointPolygonTest(conti, center, False) >= 0:
                            cv2.drawContours(np_patch, [conti], 0, 255, -1)

                np_patch[blackpart == 255] = 0
                #e_kernel = np.ones((3,3),np.uint8)
                e_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                patch_segmented = cv2.erode(np_patch, e_kernel, iterations=iter)

                im = np.copy(img0)
                
                im[np_patch == 255] = (255, 0, 0)
                im[np_core == 255] = (0, 255, 0)

                _, patch_cont, _ = cv2.findContours(patch_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                patch_assignment = []
                patch_angle = []
                for conti in patch_cont:
                    # cv2.circle(im,(conti[0][0][0],conti[0][0][1]),5,(0,0,255),-1)
                    # print (len(conti))

                    # patchi = np.copy(canvas)
                    # cv2.drawContours(patchi,[conti],0,255,1)
                    # min_dist, _, _, _ = cv2.minMaxLoc(distance_map, mask=patchi)
                    core_attached = []
                    contact_angle = []
                    # for i,mapi in enumerate(distance_map):
                    # min_dist,_,min_loc,_ = cv2.minMaxLoc(mapi,mask = patchi)
                    # if min_dist < 10:
                    # core_attached.append(i)
                    # cv2.circle(im,min_loc,1,(0,0,255),-1)
                    # patch_assignment.append(core_attached)
                    for i, npi in enumerate(np_core_list):
                        npi_flag = False
                        corner = []
                        start_index = -1
                        l = 0
                        for j, pointj in enumerate(conti):
                            x = pointj[0][0]
                            y = pointj[0][1]
                            if np.any(npi[y - iter * 3:y + iter * 3, x - iter * 3:x + iter * 3] == 255) and np.any(
                                    graypart[y - iter * 3:y + iter * 3, x - iter * 3:x + iter * 3] == 0):
                                npi_flag = True
                                if start_index == -1:
                                    start_index = j
                                l += 1
                            elif start_index != -1:
                                corner.append([start_index, l])
                                l = 0
                                start_index = -1

                        if npi_flag == True and len(corner) > 0:
                            # print (corner)
                            core_attached.append(i)

                            corner1 = corner[-1][0] + int(corner[-1][1] / 2)
                            if l == 0:
                                corner2 = corner[0][0] + int(corner[0][1] / 2)
                            elif len(corner) == 2:
                                corner2 = (start_index + int((l + corner[0][0]) / 2)) % len(conti)
                            else:
                                corner2 = start_index + int(l / 2)

                            eq_l = int(cv2.contourArea(conti) ** 0.5 / 2)
                            corner1_pre = (corner1 - eq_l) % len(conti)
                            corner1_aft = (corner1 + eq_l) % len(conti)
                            corner2_pre = (corner2 - eq_l) % len(conti)
                            corner2_aft = (corner2 + eq_l) % len(conti)
                            p1 = (conti[corner1][0][0], conti[corner1][0][1])
                            p1_pre = (conti[corner1_pre][0][0], conti[corner1_pre][0][1])

                            p1_aft = (conti[corner1_aft][0][0], conti[corner1_aft][0][1])

                            p2 = (conti[corner2][0][0], conti[corner2][0][1])
                            p2_pre = (conti[corner2_pre][0][0], conti[corner2_pre][0][1])

                            p2_aft = (conti[corner2_aft][0][0], conti[corner2_aft][0][1])

                            angle1 = CalcAngle(p1, p1_aft, p1_pre) * 180 / np.pi
                            angle2 = CalcAngle(p2, p2_aft, p2_pre) * 180 / np.pi
                            # print (angle1,angle2)
                            contact_angle.append((angle1, angle2))

                            # cv2.circle(im,p1,2,(0,0,255),-1)
                            # cv2.circle(im,p2,2,(0,0,255),-1)
                            cv2.line(im, p1, p1_pre, (0, 0, 255), 3)
                            cv2.line(im, p1, p1_aft, (0, 0, 255), 3)
                            cv2.line(im, p2, p2_pre, (0, 0, 255), 3)
                            cv2.line(im, p2, p2_aft, (0, 0, 255), 3)

                    patch_assignment.append(core_attached)
                    patch_angle.append(contact_angle)

                # print (patch_assignment)
                # print (len(patch_cont),len(patch_assignment))

                patch_size = []
                patch_aff = []
                patch_ang = []
                for i, patchi in enumerate(patch_assignment):
                    area = cv2.contourArea(patch_cont[i]) + iter * cv2.arcLength(patch_cont[i],True)
                    if patchi == []:
                        continue
                    for j, corej in enumerate(patchi):
                        patch_size.append(area / len(patchi) * mag ** 2)
                        patch_aff.append(corej)
                        patch_ang.append((patch_angle[i][j][0], patch_angle[i][j][1]))

                # print(patch_ang)
                # print(patch_aff)

                # for i,patchi in enumerate(patch_assignment):
                #    area = cv2.contourArea(patch_cont[i])
                #    if area > 0:
                #        #print (patch_cont[i])
                #        for point in patch_cont[i]:
                #            cv2.circle(im,(point[0][0],point[0][1]),1,(0,0,255),-1)

                for i, centeri in enumerate(np_center_list):
                    cv2.putText(im, str(i), centeri, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

            im_resize = cv2.resize(im, (c2, r2), interpolation=cv2.INTER_AREA)
            patch_segmented_resize = cv2.resize(patch_segmented, (c2, r2), interpolation=cv2.INTER_AREA)
            if show_flag == 0:
                cv2.imshow('dst', im_resize)
            elif show_flag == 1:
                cv2.imshow('dst', patch_segmented_resize)
            else:
                cv2.imshow('dst', img_resize)

            tl_flag = thresh_low
            th_flag = thresh_high

        ExportData()
        cv2.imwrite('./results/p' + imgfile, im)
        cv2.imwrite('./results/m' + imgfile, patch_segmented)
        cv2.destroyAllWindows()
