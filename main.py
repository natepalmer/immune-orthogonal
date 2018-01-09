# Identify highly immunogenic peptides from orthologs to design multiple-round therapy to minimize cross-reactivity.

from Bio import SeqIO
from subprocess import run
import numpy as np
import shlex
import os
import sys


# Set of all HLA alleles used in analysis. For HLA-2, all combinations of DPA and DPB were used.
# Similarly, we used all combinations of DQA and DQB.

HLA_1_ALLELES = "HLA-A0101,HLA-A0201,HLA-A0202,HLA-A0203,HLA-A0205,HLA-A0206,HLA-A0207,HLA-A0211,HLA-A0212," \
	"HLA-A0216,HLA-A0217,HLA-A0219,HLA-A0250,HLA-A0301,HLA-A1101,HLA-A2301,HLA-A2402,HLA-A2403,HLA-A2501,HLA-A2601," \
	"HLA-A2602,HLA-A2603,HLA-A2902,HLA-A3001,HLA-A3002,HLA-A3101,HLA-A3201,HLA-A3207,HLA-A3215,HLA-A3301,HLA-A6601," \
	"HLA-A6801,HLA-A6802,HLA-A6823,HLA-A6901,HLA-A8001,HLA-B0702,HLA-B0801,HLA-B0802,HLA-B0803,HLA-B1402,HLA-B1501," \
	"HLA-B1502,HLA-B1503,HLA-B1509,HLA-B1517,HLA-B1801,HLA-B2705,HLA-B2720,HLA-B3501,HLA-B3503,HLA-B3801,HLA-B3901," \
	"HLA-B4001,HLA-B4002,HLA-B4013,HLA-B4201,HLA-B4402,HLA-B4403,HLA-B4501,HLA-B4601,HLA-B4801,HLA-B5101,HLA-B5301," \
	"HLA-B5401,HLA-B5701,HLA-B5801,HLA-B5802,HLA-B7301,HLA-B8301,HLA-C0303,HLA-C0401,HLA-C0501,HLA-C0602,HLA-C0701," \
	"HLA-C0702,HLA-C0802,HLA-C1203,HLA-C1402,HLA-C1502,HLA-E0101"

HLA_2_DR = ['DRB1_0101', 'DRB1_0102', 'DRB1_0103', 'DRB1_0104', 'DRB1_0105', 'DRB1_0106', 'DRB1_0107', 'DRB1_0108',
			'DRB1_0109', 'DRB1_0110', 'DRB1_0111', 'DRB1_0112', 'DRB1_0113', 'DRB1_0114', 'DRB1_0115', 'DRB1_0116',
			'DRB1_0117', 'DRB1_0118', 'DRB1_0119', 'DRB1_0120', 'DRB1_0121', 'DRB1_0122', 'DRB1_0123', 'DRB1_0124',
			'DRB1_0125', 'DRB1_0126', 'DRB1_0127', 'DRB1_0128', 'DRB1_0129', 'DRB1_0130', 'DRB1_0131', 'DRB1_0132',
			'DRB1_0301', 'DRB1_0302', 'DRB1_0303', 'DRB1_0304', 'DRB1_0305', 'DRB1_0306', 'DRB1_0307', 'DRB1_0308',
			'DRB1_0310', 'DRB1_0311', 'DRB1_0313', 'DRB1_0314', 'DRB1_0315', 'DRB1_0317', 'DRB1_0318', 'DRB1_0319',
			'DRB1_0320', 'DRB1_0321', 'DRB1_0322', 'DRB1_0323', 'DRB1_0324', 'DRB1_0325', 'DRB1_0326', 'DRB1_0327',
			'DRB1_0328', 'DRB1_0329', 'DRB1_0330', 'DRB1_0331', 'DRB1_0332', 'DRB1_0333', 'DRB1_0334', 'DRB1_0335',
			'DRB1_0336', 'DRB1_0337', 'DRB1_0338', 'DRB1_0339', 'DRB1_0340', 'DRB1_0341', 'DRB1_0342', 'DRB1_0343',
			'DRB1_0344', 'DRB1_0345', 'DRB1_0346', 'DRB1_0347', 'DRB1_0348', 'DRB1_0349', 'DRB1_0350', 'DRB1_0351',
			'DRB1_0352', 'DRB1_0353', 'DRB1_0354', 'DRB1_0355', 'DRB1_0401', 'DRB1_0402', 'DRB1_0403', 'DRB1_0404',
			'DRB1_0405', 'DRB1_0406', 'DRB1_0407', 'DRB1_0408', 'DRB1_0409', 'DRB1_0410', 'DRB1_0411', 'DRB1_0412',
			'DRB1_0413', 'DRB1_0414', 'DRB1_0415', 'DRB1_0416', 'DRB1_0417', 'DRB1_0418', 'DRB1_0419', 'DRB1_0421',
			'DRB1_0422', 'DRB1_0423', 'DRB1_0424', 'DRB1_0426', 'DRB1_0427', 'DRB1_0428', 'DRB1_0429', 'DRB1_0430',
			'DRB1_0431', 'DRB1_0433', 'DRB1_0434', 'DRB1_0435', 'DRB1_0436', 'DRB1_0437', 'DRB1_0438', 'DRB1_0439',
			'DRB1_0440', 'DRB1_0441', 'DRB1_0442', 'DRB1_0443', 'DRB1_0444', 'DRB1_0445', 'DRB1_0446', 'DRB1_0447',
			'DRB1_0448', 'DRB1_0449', 'DRB1_0450', 'DRB1_0451', 'DRB1_0452', 'DRB1_0453', 'DRB1_0454', 'DRB1_0455',
			'DRB1_0456', 'DRB1_0457', 'DRB1_0458', 'DRB1_0459', 'DRB1_0460', 'DRB1_0461', 'DRB1_0462', 'DRB1_0463',
			'DRB1_0464', 'DRB1_0465', 'DRB1_0466', 'DRB1_0467', 'DRB1_0468', 'DRB1_0469', 'DRB1_0470', 'DRB1_0471',
			'DRB1_0472', 'DRB1_0473', 'DRB1_0474', 'DRB1_0475', 'DRB1_0476', 'DRB1_0477', 'DRB1_0478', 'DRB1_0479',
			'DRB1_0480', 'DRB1_0482', 'DRB1_0483', 'DRB1_0484', 'DRB1_0485', 'DRB1_0486', 'DRB1_0487', 'DRB1_0488',
			'DRB1_0489', 'DRB1_0491', 'DRB1_0701', 'DRB1_0703', 'DRB1_0704', 'DRB1_0705', 'DRB1_0706', 'DRB1_0707',
			'DRB1_0708', 'DRB1_0709', 'DRB1_0711', 'DRB1_0712', 'DRB1_0713', 'DRB1_0714', 'DRB1_0715', 'DRB1_0716',
			'DRB1_0717', 'DRB1_0719', 'DRB1_0801', 'DRB1_0802', 'DRB1_0803', 'DRB1_0804', 'DRB1_0805', 'DRB1_0806',
			'DRB1_0807', 'DRB1_0808', 'DRB1_0809', 'DRB1_0810', 'DRB1_0811', 'DRB1_0812', 'DRB1_0813', 'DRB1_0814',
			'DRB1_0815', 'DRB1_0816', 'DRB1_0818', 'DRB1_0819', 'DRB1_0820', 'DRB1_0821', 'DRB1_0822', 'DRB1_0823',
			'DRB1_0824', 'DRB1_0825', 'DRB1_0826', 'DRB1_0827', 'DRB1_0828', 'DRB1_0829', 'DRB1_0830', 'DRB1_0831',
			'DRB1_0832', 'DRB1_0833', 'DRB1_0834', 'DRB1_0835', 'DRB1_0836', 'DRB1_0837', 'DRB1_0838', 'DRB1_0839',
			'DRB1_0840', 'DRB1_0901', 'DRB1_0902', 'DRB1_0903', 'DRB1_0904', 'DRB1_0905', 'DRB1_0906', 'DRB1_0907',
			'DRB1_0908', 'DRB1_0909', 'DRB1_1001', 'DRB1_1002', 'DRB1_1003', 'DRB1_1101', 'DRB1_1102', 'DRB1_1103',
			'DRB1_1104', 'DRB1_1105', 'DRB1_1106', 'DRB1_1107', 'DRB1_1108', 'DRB1_1109', 'DRB1_1110', 'DRB1_1111',
			'DRB1_1112', 'DRB1_1113', 'DRB1_1114', 'DRB1_1115', 'DRB1_1116', 'DRB1_1117', 'DRB1_1118', 'DRB1_1119',
			'DRB1_1120', 'DRB1_1121', 'DRB1_1124', 'DRB1_1125', 'DRB1_1127', 'DRB1_1128', 'DRB1_1129', 'DRB1_1130',
			'DRB1_1131', 'DRB1_1132', 'DRB1_1133', 'DRB1_1134', 'DRB1_1135', 'DRB1_1136', 'DRB1_1137', 'DRB1_1138',
			'DRB1_1139', 'DRB1_1141', 'DRB1_1142', 'DRB1_1143', 'DRB1_1144', 'DRB1_1145', 'DRB1_1146', 'DRB1_1147',
			'DRB1_1148', 'DRB1_1149', 'DRB1_1150', 'DRB1_1151', 'DRB1_1152', 'DRB1_1153', 'DRB1_1154', 'DRB1_1155',
			'DRB1_1156', 'DRB1_1157', 'DRB1_1158', 'DRB1_1159', 'DRB1_1160', 'DRB1_1161', 'DRB1_1162', 'DRB1_1163',
			'DRB1_1164', 'DRB1_1165', 'DRB1_1166', 'DRB1_1167', 'DRB1_1168', 'DRB1_1169', 'DRB1_1170', 'DRB1_1172',
			'DRB1_1173', 'DRB1_1174', 'DRB1_1175', 'DRB1_1176', 'DRB1_1177', 'DRB1_1178', 'DRB1_1179', 'DRB1_1180',
			'DRB1_1181', 'DRB1_1182', 'DRB1_1183', 'DRB1_1184', 'DRB1_1185', 'DRB1_1186', 'DRB1_1187', 'DRB1_1188',
			'DRB1_1189', 'DRB1_1190', 'DRB1_1191', 'DRB1_1192', 'DRB1_1193', 'DRB1_1194', 'DRB1_1195', 'DRB1_1196',
			'DRB1_1201', 'DRB1_1202', 'DRB1_1203', 'DRB1_1204', 'DRB1_1205', 'DRB1_1206', 'DRB1_1207', 'DRB1_1208',
			'DRB1_1209', 'DRB1_1210', 'DRB1_1211', 'DRB1_1212', 'DRB1_1213', 'DRB1_1214', 'DRB1_1215', 'DRB1_1216',
			'DRB1_1217', 'DRB1_1218', 'DRB1_1219', 'DRB1_1220', 'DRB1_1221', 'DRB1_1222', 'DRB1_1223', 'DRB1_1301',
			'DRB1_1302', 'DRB1_1303', 'DRB1_1304', 'DRB1_1305', 'DRB1_1306', 'DRB1_1307', 'DRB1_1308', 'DRB1_1309',
			'DRB1_1310', 'DRB1_13100', 'DRB1_13101', 'DRB1_1311', 'DRB1_1312', 'DRB1_1313', 'DRB1_1314', 'DRB1_1315',
			'DRB1_1316', 'DRB1_1317', 'DRB1_1318', 'DRB1_1319', 'DRB1_1320', 'DRB1_1321', 'DRB1_1322', 'DRB1_1323',
			'DRB1_1324', 'DRB1_1326', 'DRB1_1327', 'DRB1_1329', 'DRB1_1330', 'DRB1_1331', 'DRB1_1332', 'DRB1_1333',
			'DRB1_1334', 'DRB1_1335', 'DRB1_1336', 'DRB1_1337', 'DRB1_1338', 'DRB1_1339', 'DRB1_1341', 'DRB1_1342',
			'DRB1_1343', 'DRB1_1344', 'DRB1_1346', 'DRB1_1347', 'DRB1_1348', 'DRB1_1349', 'DRB1_1350', 'DRB1_1351',
			'DRB1_1352', 'DRB1_1353', 'DRB1_1354', 'DRB1_1355', 'DRB1_1356', 'DRB1_1357', 'DRB1_1358', 'DRB1_1359',
			'DRB1_1360', 'DRB1_1361', 'DRB1_1362', 'DRB1_1363', 'DRB1_1364', 'DRB1_1365', 'DRB1_1366', 'DRB1_1367',
			'DRB1_1368', 'DRB1_1369', 'DRB1_1370', 'DRB1_1371', 'DRB1_1372', 'DRB1_1373', 'DRB1_1374', 'DRB1_1375',
			'DRB1_1376', 'DRB1_1377', 'DRB1_1378', 'DRB1_1379', 'DRB1_1380', 'DRB1_1381', 'DRB1_1382', 'DRB1_1383',
			'DRB1_1384', 'DRB1_1385', 'DRB1_1386', 'DRB1_1387', 'DRB1_1388', 'DRB1_1389', 'DRB1_1390', 'DRB1_1391',
			'DRB1_1392', 'DRB1_1393', 'DRB1_1394', 'DRB1_1395', 'DRB1_1396', 'DRB1_1397', 'DRB1_1398', 'DRB1_1399',
			'DRB1_1401', 'DRB1_1402', 'DRB1_1403', 'DRB1_1404', 'DRB1_1405', 'DRB1_1406', 'DRB1_1407', 'DRB1_1408',
			'DRB1_1409', 'DRB1_1410', 'DRB1_1411', 'DRB1_1412', 'DRB1_1413', 'DRB1_1414', 'DRB1_1415', 'DRB1_1416',
			'DRB1_1417', 'DRB1_1418', 'DRB1_1419', 'DRB1_1420', 'DRB1_1421', 'DRB1_1422', 'DRB1_1423', 'DRB1_1424',
			'DRB1_1425', 'DRB1_1426', 'DRB1_1427', 'DRB1_1428', 'DRB1_1429', 'DRB1_1430', 'DRB1_1431', 'DRB1_1432',
			'DRB1_1433', 'DRB1_1434', 'DRB1_1435', 'DRB1_1436', 'DRB1_1437', 'DRB1_1438', 'DRB1_1439', 'DRB1_1440',
			'DRB1_1441', 'DRB1_1442', 'DRB1_1443', 'DRB1_1444', 'DRB1_1445', 'DRB1_1446', 'DRB1_1447', 'DRB1_1448',
			'DRB1_1449', 'DRB1_1450', 'DRB1_1451', 'DRB1_1452', 'DRB1_1453', 'DRB1_1454', 'DRB1_1455', 'DRB1_1456',
			'DRB1_1457', 'DRB1_1458', 'DRB1_1459', 'DRB1_1460', 'DRB1_1461', 'DRB1_1462', 'DRB1_1463', 'DRB1_1464',
			'DRB1_1465', 'DRB1_1467', 'DRB1_1468', 'DRB1_1469', 'DRB1_1470', 'DRB1_1471', 'DRB1_1472', 'DRB1_1473',
			'DRB1_1474', 'DRB1_1475', 'DRB1_1476', 'DRB1_1477', 'DRB1_1478', 'DRB1_1479', 'DRB1_1480', 'DRB1_1481',
			'DRB1_1482', 'DRB1_1483', 'DRB1_1484', 'DRB1_1485', 'DRB1_1486', 'DRB1_1487', 'DRB1_1488', 'DRB1_1489',
			'DRB1_1490', 'DRB1_1491', 'DRB1_1493', 'DRB1_1494', 'DRB1_1495', 'DRB1_1496', 'DRB1_1497', 'DRB1_1498',
			'DRB1_1499', 'DRB1_1501', 'DRB1_1502', 'DRB1_1503', 'DRB1_1504', 'DRB1_1505', 'DRB1_1506', 'DRB1_1507',
			'DRB1_1508', 'DRB1_1509', 'DRB1_1510', 'DRB1_1511', 'DRB1_1512', 'DRB1_1513', 'DRB1_1514', 'DRB1_1515',
			'DRB1_1516', 'DRB1_1518', 'DRB1_1519', 'DRB1_1520', 'DRB1_1521', 'DRB1_1522', 'DRB1_1523', 'DRB1_1524',
			'DRB1_1525', 'DRB1_1526', 'DRB1_1527', 'DRB1_1528', 'DRB1_1529', 'DRB1_1530', 'DRB1_1531', 'DRB1_1532',
			'DRB1_1533', 'DRB1_1534', 'DRB1_1535', 'DRB1_1536', 'DRB1_1537', 'DRB1_1538', 'DRB1_1539', 'DRB1_1540',
			'DRB1_1541', 'DRB1_1542', 'DRB1_1543', 'DRB1_1544', 'DRB1_1545', 'DRB1_1546', 'DRB1_1547', 'DRB1_1548',
			'DRB1_1549', 'DRB1_1601', 'DRB1_1602', 'DRB1_1603', 'DRB1_1604', 'DRB1_1605', 'DRB1_1607', 'DRB1_1608',
			'DRB1_1609', 'DRB1_1610', 'DRB1_1611', 'DRB1_1612', 'DRB1_1614', 'DRB1_1615', 'DRB1_1616', 'DRB3_0101',
			'DRB3_0104', 'DRB3_0105', 'DRB3_0108', 'DRB3_0109', 'DRB3_0111', 'DRB3_0112', 'DRB3_0113', 'DRB3_0114',
			'DRB3_0201', 'DRB3_0202', 'DRB3_0204', 'DRB3_0205', 'DRB3_0209', 'DRB3_0210', 'DRB3_0211', 'DRB3_0212',
			'DRB3_0213', 'DRB3_0214', 'DRB3_0215', 'DRB3_0216', 'DRB3_0217', 'DRB3_0218', 'DRB3_0219', 'DRB3_0220',
			'DRB3_0221', 'DRB3_0222', 'DRB3_0223', 'DRB3_0224', 'DRB3_0225', 'DRB3_0301', 'DRB3_0303', 'DRB4_0101',
			'DRB4_0103', 'DRB4_0104', 'DRB4_0106', 'DRB4_0107', 'DRB4_0108', 'DRB5_0101', 'DRB5_0102', 'DRB5_0103',
			'DRB5_0104', 'DRB5_0105', 'DRB5_0106', 'DRB5_0108N', 'DRB5_0111', 'DRB5_0112', 'DRB5_0113', 'DRB5_0114',
			'DRB5_0202', 'DRB5_0203', 'DRB5_0204', 'DRB5_0205']

HLA_2_DPA = ['DPA1_0103', 'DPA1_0104', 'DPA1_0105', 'DPA1_0106', 'DPA1_0107', 'DPA1_0108', 'DPA1_0109', 'DPA1_0110',
				 'DPA1_0201', 'DPA1_0202', 'DPA1_0203', 'DPA1_0204', 'DPA1_0301', 'DPA1_0302', 'DPA1_0303', 'DPA1_0401']

HLA_2_DPB = ['DPB1_0101', 'DPB1_0201', 'DPB1_0202', 'DPB1_0301', 'DPB1_0401', 'DPB1_0402', 'DPB1_0501',
				 'DPB1_0601', 'DPB1_0801', 'DPB1_0901', 'DPB1_10001', 'DPB1_1001', 'DPB1_10101', 'DPB1_10201',
				 'DPB1_10301', 'DPB1_10401', 'DPB1_10501', 'DPB1_10601', 'DPB1_10701', 'DPB1_10801', 'DPB1_10901',
				 'DPB1_11001', 'DPB1_1101', 'DPB1_11101', 'DPB1_11201', 'DPB1_11301', 'DPB1_11401', 'DPB1_11501',
				 'DPB1_11601', 'DPB1_11701', 'DPB1_11801', 'DPB1_11901', 'DPB1_12101', 'DPB1_12201', 'DPB1_12301',
				 'DPB1_12401', 'DPB1_12501', 'DPB1_12601', 'DPB1_12701', 'DPB1_12801', 'DPB1_12901', 'DPB1_13001',
				 'DPB1_1301', 'DPB1_13101', 'DPB1_13201', 'DPB1_13301', 'DPB1_13401', 'DPB1_1401', 'DPB1_1501',
				 'DPB1_1601', 'DPB1_1701', 'DPB1_1801', 'DPB1_1901', 'DPB1_2001', 'DPB1_2101', 'DPB1_2201',
				 'DPB1_2301', 'DPB1_2401', 'DPB1_2501', 'DPB1_2601', 'DPB1_2701', 'DPB1_2801', 'DPB1_2901',
				 'DPB1_3001', 'DPB1_3101', 'DPB1_3201', 'DPB1_3301', 'DPB1_3401', 'DPB1_3501', 'DPB1_3601',
				 'DPB1_3701', 'DPB1_3801', 'DPB1_3901', 'DPB1_4001', 'DPB1_4101', 'DPB1_4401', 'DPB1_4501',
				 'DPB1_4601', 'DPB1_4701', 'DPB1_4801', 'DPB1_4901', 'DPB1_5001', 'DPB1_5101', 'DPB1_5201',
				 'DPB1_5301', 'DPB1_5401', 'DPB1_5501', 'DPB1_5601', 'DPB1_5801', 'DPB1_5901', 'DPB1_6001',
				 'DPB1_6201', 'DPB1_6301', 'DPB1_6501', 'DPB1_6601', 'DPB1_6701', 'DPB1_6801', 'DPB1_6901',
				 'DPB1_7001', 'DPB1_7101', 'DPB1_7201', 'DPB1_7301', 'DPB1_7401', 'DPB1_7501', 'DPB1_7601',
				 'DPB1_7701', 'DPB1_7801', 'DPB1_7901', 'DPB1_8001', 'DPB1_8101', 'DPB1_8201', 'DPB1_8301',
				 'DPB1_8401', 'DPB1_8501', 'DPB1_8601', 'DPB1_8701', 'DPB1_8801', 'DPB1_8901', 'DPB1_9001',
				 'DPB1_9101', 'DPB1_9201', 'DPB1_9301', 'DPB1_9401', 'DPB1_9501', 'DPB1_9601', 'DPB1_9701',
				 'DPB1_9801', 'DPB1_9901']

HLA_2_DQA = ['DQA1_0101', 'DQA1_0102', 'DQA1_0103', 'DQA1_0104', 'DQA1_0105', 'DQA1_0106', 'DQA1_0107',
				 'DQA1_0108', 'DQA1_0109', 'DQA1_0201', 'DQA1_0301', 'DQA1_0302', 'DQA1_0303', 'DQA1_0401',
				 'DQA1_0402', 'DQA1_0404', 'DQA1_0501', 'DQA1_0503', 'DQA1_0504', 'DQA1_0505', 'DQA1_0506',
				 'DQA1_0507', 'DQA1_0508', 'DQA1_0509', 'DQA1_0510', 'DQA1_0511', 'DQA1_0601', 'DQA1_0602']

HLA_2_DQB = ['DQB1_0201', 'DQB1_0202', 'DQB1_0203', 'DQB1_0204', 'DQB1_0205', 'DQB1_0206', 'DQB1_0301',
				 'DQB1_0302', 'DQB1_0303', 'DQB1_0304', 'DQB1_0305', 'DQB1_0306', 'DQB1_0307', 'DQB1_0308',
				 'DQB1_0309', 'DQB1_0310', 'DQB1_0311', 'DQB1_0312', 'DQB1_0313', 'DQB1_0314', 'DQB1_0315',
				 'DQB1_0316', 'DQB1_0317', 'DQB1_0318', 'DQB1_0319', 'DQB1_0320', 'DQB1_0321', 'DQB1_0322',
				 'DQB1_0323', 'DQB1_0324', 'DQB1_0325', 'DQB1_0326', 'DQB1_0327', 'DQB1_0328', 'DQB1_0329',
				 'DQB1_0330', 'DQB1_0331', 'DQB1_0332', 'DQB1_0333', 'DQB1_0334', 'DQB1_0335', 'DQB1_0336',
				 'DQB1_0337', 'DQB1_0338', 'DQB1_0401', 'DQB1_0402', 'DQB1_0403', 'DQB1_0404', 'DQB1_0405',
				 'DQB1_0406', 'DQB1_0407', 'DQB1_0408', 'DQB1_0501', 'DQB1_0502', 'DQB1_0503', 'DQB1_0505',
				 'DQB1_0506', 'DQB1_0507', 'DQB1_0508', 'DQB1_0509', 'DQB1_0510', 'DQB1_0511', 'DQB1_0512',
				 'DQB1_0513', 'DQB1_0514', 'DQB1_0601', 'DQB1_0602', 'DQB1_0603', 'DQB1_0604', 'DQB1_0607',
				 'DQB1_0608', 'DQB1_0609', 'DQB1_0610', 'DQB1_0611', 'DQB1_0612', 'DQB1_0614', 'DQB1_0615',
				 'DQB1_0616', 'DQB1_0617', 'DQB1_0618', 'DQB1_0619', 'DQB1_0621', 'DQB1_0622', 'DQB1_0623',
				 'DQB1_0624', 'DQB1_0625', 'DQB1_0627', 'DQB1_0628', 'DQB1_0629', 'DQB1_0630', 'DQB1_0631',
				 'DQB1_0632', 'DQB1_0633', 'DQB1_0634', 'DQB1_0635', 'DQB1_0636', 'DQB1_0637', 'DQB1_0638',
				 'DQB1_0639', 'DQB1_0640', 'DQB1_0641', 'DQB1_0642', 'DQB1_0643', 'DQB1_0644']


class Peptide:
    """Class used to store MHC binding attributes of a peptide as predicted by netMHC"""

    def __init__(self, string, MHC):
        attributes = string.split()
        self.binding = None

        if MHC in (1, "1"):
            binding_length = 16
            self.core = attributes[3]
            self.offset = int(attributes[4])
            self.insertion_position = int(attributes[5])
            self.insertion_length = int(attributes[6])
            self.deletion_position = int(attributes[7])
            self.deletion_length = int(attributes[8])
            self.interaction_core = attributes[9]
            self.protein = attributes[10]
            self.pAffinity = float(attributes[11])
            self.nM_affinity = float(attributes[12])
            self.percentile_rank = float(attributes[13])

        elif MHC in (2, "2"):
            binding_length = 12
            self.core = attributes[5]
            self.offset = int(attributes[4])
            self.protein = attributes[3]
            self.pAffinity = float(attributes[7])
            self.nM_affinity = float(attributes[8])
            self.percentile_rank = float(attributes[9])

        if len(attributes) == binding_length:
            if "S" in attributes[-1]:
                self.binding = 'strong'
            elif "W" in attributes[-1]:
                self.binding = 'weak'

        self.position = int(attributes[0])
        self.HLA = attributes[1]
        self.seq = attributes[2]

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return "{0} peptide {1}. HLA {2}, affinity {3} nM, percentile {4}".format(
            self.protein, self.seq, self.HLA, self.nM_affinity, self.percentile_rank)

    def __str__(self):
        return repr(self)


class netMHC_prediction:
    """Class used to store total MHC binding prediction information for a protein.
    Main attribute is self.peptides which is a list of Peptide objects comprising
    all peptides in a protein and their binding information for a given MHC allele."""

    def __init__(self, filename, MHC):
        self.metadata = ""
        self.peptides = list()
        with open(filename) as inputfile:
            print("Parsing netMHC file {}".format(filename))
            for line in inputfile.readlines():
                x = line.strip()
                if len(x) == 0:
                    continue
                if x[0] == "#":
                    self.metadata += "\n" + x
                if x[0].isdigit():
                    self.peptides.append(Peptide(x, MHC))

    def get_binders(self, threshold=0.5, affinity=False):
        if affinity:
            return [p for p in self.peptides if p.nM_affinity < affinity]
        else:
            return [p for p in self.peptides if p.percentile_rank < threshold]

    def __len__(self):
        return len(self.peptides)

    def __str__(self):
        return "netMHC prediction"


def count(seq, reference, mismatches=0, positions=False, quick=True):
    """Count the number of appearances of substring :seq in string :reference with at most :mismatches.
    If positions flag is True, return a list of the locations of seq in reference."""
    
    if quick:
    	return int(seq in reference)
    
    n = len(seq)
    overlap_locations = list()
    for i in range(len(reference) - n + 1):
        score = 0
        for j in range(n):
            if seq[j] == reference[i + j]:
                score += 1
        if n - score <= mismatches:
            overlap_locations.append(i)
    if positions:
        return overlap_locations
    return len(overlap_locations)


def compare_peptides(prediction, seq, threshold=0.5, core=False):
    total_overlap = 0
    for p in prediction.get_binders(threshold=threshold):
        if type(seq) == str:
            s = seq
        else:
            s = str(seq.seq)
        if core:
            total_overlap += count(p.core, s)
        else:
            total_overlap += count(p.seq, s)
    return total_overlap


def build_MHCI_matrix(fasta, prediction_dir, threshold=2, run_missing_predictions=True):
    """Build immune overlap matrix of all proteins in :fasta.
    Peptides are classified as binding if affinity is in the top :threshold percentile. Will use exiting predictions if
    available, if not and :run_missing_predictions flag is True (default) then will call netMHC-4.0 to make predictions.
    If no prediction can be found and :run_missing_predictions flag is False, raises an exception."""

    proteins = [x for x in SeqIO.parse(fasta, "fasta")]
    overlap = np.zeros((len(proteins), len(proteins)))
    predictions = dict()

    for i in range(len(proteins)):

        # Write protein to its own temporary fasta file for prediction input
        with open("single.fasta", mode="w") as single:
            print(proteins[i].format("fasta"), file=single)

        # Look for existing prediction
        prediction_fn = prediction_dir+"/{}.tsv".format(proteins[i].name)
        if not os.path.isfile(prediction_fn):
            if run_missing_predictions == False:
                raise Exception("Can't find prediction file and run_missing_predictions flag is False. Check that "
                                "predictions are in directory MHCI_predictions and that folder is in the current path.")

            # Run netMCH-4.0 on single protein
            print("Running netMHC on {}".format(proteins[i].name))
            with open(prediction_fn, mode="w") as predict_file:
                run(shlex.split("/Applications/netMHC-4.0/netMHC " \
                                "-a {0} " \
                                "-t 0.1 " \
                                "-s " \
                                "-l 8,9,10,11 " \
                                "-f single.fasta".format(HLA_1_ALLELES)), stdout=predict_file)

        # Parse prediction file
        predictions[proteins[i].name] = netMHC_prediction(prediction_fn, 1)

        # Compare set of immunogenic (defined by :threshold) peptides to all other protein sequence
        print("Comparing top {}% immunogenic peptides to other proteins".format(threshold))
        for j in range(i + 1, len(proteins)):
            overlap[i, j] = compare_peptides(predictions[proteins[i].name], proteins[j], threshold=threshold)
            print(".", end="")

        run(["rm", "single.fasta"])
        print("Done")

    return overlap


def build_MHCII_matrix(fasta, prediction_dir, threshold=2, run_missing_predictions=True):
    """Build immune overlap matrix of all proteins in :fasta.
    Peptides are classified as binding if affinity is in the top :threshold percentile. Will use exiting predictions if
    available, if not and :run_missing_predictions flag is True (default) then will call netMHCIIpan-3.1 to make predictions.
    If no prediction can be found and :run_missing_predictions flag is False, raises an exception."""

    proteins = [x for x in SeqIO.parse(fasta, "fasta")]
    overlap = np.zeros((len(proteins), len(proteins)))
    predictions = dict()

    for i in range(len(proteins)):

        # Write protein to its own fasta file for netMHCpan input
        with open("single.fasta", mode="w") as single:
            print(proteins[i].format("fasta"), file=single)

        # Look for existing predictions
        prediction_fn = prediction_dir+"/{}.tsv".format(proteins[i].name)
        if not os.path.isfile(prediction_fn):
            if run_missing_predictions == False:
                raise Exception("Can't find prediction file and run_missing_predictions flag is False. Check that "
                                "predictions are in directory MHCII_predictions and that folder is in the current path.")

            # Run netMHCIIpan on single protein
            print("Running netMHCIIpan on {}".format(proteins[i].name))
            run(shlex.split("touch {}".format(prediction_fn)))
            with open(prediction_fn, mode="w") as predict_file:
                run(shlex.split("/Applications/netMHCIIpan-3.1/netMHCIIpan " \
                                "-a {0} " \
                                "-u " \
                                "-s " \
                                "-f single.fasta".format("DRB1_1501")), stdout=predict_file)

        # Parse prediction file
        predictions[proteins[i].name] = netMHC_prediction(prediction_fn, 2)

        # Compare set of immunogenic (defined by :threshold) peptides to all other protein sequence
        print("Comparing top {}% immunogenic peptides to other proteins".format(threshold))
        for j in range(i + 1, len(proteins)):
            overlap[i, j] = compare_peptides(predictions[proteins[i].name], proteins[j], threshold=threshold)
            print(".", end="")

        run(["rm", "single.fasta"])
        print("Done")

    return overlap


def direct_peptide(p1, p2, core=True, threshold=2):
    """Compare peptides in prediction object :p1 to those in prediction object :p2.
    This is equivalent, but slower than comparing the peptides in :p1 to the parent sequence for object :p2"""
    b1 = p1.get_binders(threshold=threshold)
    b2 = p2.get_binders(threshold=threshold)
    if core:
        test = [y.core for y in b2]
        overlap = sum([x.core in test for x in b1])
    else:
        overlap = sum([x in b2 for x in b1])
    return overlap


if __name__ == "__main__":

    fn = sys.argv[1]
    base = fn.split(sep=".")[0]
    test_matrix = build_MHCI_matrix(fn)
    # print(test_matrix)
    np.savetxt(base + "_matrix.csv", test_matrix, delimiter=',', fmt='%10.1f')
